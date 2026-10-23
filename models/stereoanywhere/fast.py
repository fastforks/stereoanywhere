"""Inference path for Stereo Anywhere (general: any input size, no per-size compilation / tuning).

    prepare_inference(model, cfg)          # once
    disp = forward_fast(model, L, R, mono_l, mono_r, iters=32)

- conv weights cast to fp16 once; network runs under fp16 autocast (fp32 stays where autocast keeps it: softmax,
  disparity / confidence estimation, least squares, flow state)
- norms keep the fp16 dtype (autocast would run instance_norm in fp32 and return fp32 -> the whole 3D hourglass on
  the W x H x W mono volume ran fp32): statistics in fp32, normalization in place in fp16; BatchNorm (eval) as affine
- activations in place
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _instance_norm(self, x):
    dims = tuple(range(2, x.dim()))
    var, mean = torch.var_mean(x, dim=dims, keepdim=True, unbiased=False)  # fp32 accumulation for fp16 input
    rstd = torch.rsqrt(var.float() + self.eps)
    if x.dtype == torch.float16:
        return x.sub_(mean).mul_(rstd.to(x.dtype)) if not x.requires_grad else (x - mean) * rstd.to(x.dtype)
    return (x - mean) * rstd


def _batch_norm(self, x):
    shp = (1, -1) + (1,) * (x.dim() - 2)
    return x * self._s.to(x.dtype).view(shp) + self._t.to(x.dtype).view(shp)


@torch.no_grad()
def prepare_inference(model, cfg=()):
    model.eval()
    n = 0
    vol32 = set()
    cfg = set(cfg)
    if "cmix" in cfg:  # context encoder mixed precision: fp32 first conv + fp32 output heads, rest fp16 (cnet 163 -> 74 ms
        # at Midd-H vs all-fp32 "f32cnet"; ETH3D 0.239 / 1.68 = fp32 cnet; fp16 heads: EPE 0.243, bf16 cnet: 0.250 / 1.94)
        cfg |= {"c1f32", "cf32_outputs08", "cf32_outputs16", "cf32_outputs32"}
    if "c1f32" in cfg:
        # context encoder input = mono depth in [0, 1], replicated to 3 channels: in fp16 (ulp 2^-11 near 1) the small depth
        # steps between neighbouring pixels are quantized away (ETH3D D1 +0.1 with an fp16 cnet, worse in bf16). Only the
        # first conv reads it: run it in fp32 on one channel with the input-channel-summed kernel, rest of cnet fp16.
        c1 = model.cnet.conv1
        c1._w1 = c1.weight.detach().float().sum(1, keepdim=True)
        c1._b1 = c1.bias.detach().float()
        c1._cl = "cl" in cfg
        c1.forward = _conv1_f32.__get__(c1)
        vol32.add(id(c1))
    for c in cfg:  # debug: cf32_<attr> keeps model.cnet.<attr> in fp32
        if c.startswith("cf32_"):
            sub = getattr(model.cnet, c[5:])
            vol32 |= set(id(x) for x in sub.modules())
            for m in (sub if isinstance(sub, nn.ModuleList) else [sub]):
                m.forward = _fp32_wrap(m.forward)
    if "hg32" in cfg:  # debug / accuracy option: mono-volume hourglass + classifiers in fp32
        for mod in (model.hourglass_mono, model.classifier_mono, model.classifier_monoconf):
            vol32 |= set(id(x) for x in mod.modules())
    keep32 = [c[3:] for c in cfg if c.startswith("f32")]  # debug / accuracy option: top-level modules kept fp32
    for name in keep32:
        mod = getattr(model, name)
        vol32 |= set(id(x) for x in mod.modules())
        mod.forward = _fp32_wrap(mod.forward)
    keep16 = [c[3:] for c in cfg if c.startswith("b16")]  # top-level modules in bf16 (fp32 range, fp32 accumulation)
    for name in keep16:
        mod = getattr(model, name)
        vol32 |= set(id(x) for x in mod.modules())
        for m in mod.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.to(torch.bfloat16)
                if "cl" in cfg:
                    m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
        mod.forward = _bf16_wrap(mod.forward)
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm2d, nn.InstanceNorm3d)) and not m.affine and not m.track_running_stats:
            m.forward = _instance_norm.__get__(m); n += 1
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
            m._s = m.weight.float() * torch.rsqrt(m.running_var.float() + m.eps)
            m._t = m.bias.float() - m.running_mean.float() * m._s
            m.forward = _batch_norm.__get__(m); n += 1
        elif isinstance(m, (nn.ReLU, nn.LeakyReLU)):
            m.inplace = True
        elif isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            if id(m) in vol32:
                continue
            m.half()
            if "cl" in cfg and isinstance(m, nn.Conv2d):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
    if "hacc" in cfg or "gru" in cfg:
        for m in model.modules():
            if isinstance(m, ConvGRUType()):
                m._wzr = torch.cat([m.convz.weight, m.convr.weight], 0)
                m._bzr = torch.cat([m.convz.bias, m.convr.bias], 0)
                if "cl" in cfg:
                    m._wzr = m._wzr.contiguous(memory_format=torch.channels_last)
                m.forward = _gru_forward.__get__(m)
    if "gru" in cfg:
        from .update import BasicMotionEncoder
        for m in model.modules():
            if isinstance(m, BasicMotionEncoder):
                m.forward = _motion_forward.__get__(m)
    if "hacc" in cfg:
        assert "cl" in cfg
        for m in model.modules():
            if isinstance(m, nn.Conv2d) and id(m) not in vol32 and m.groups == 1 and m.dilation == (1, 1) and m.padding_mode == "zeros":
                m.forward = _conv_hacc_forward.__get__(m)
                m._hacc = True
    if "gk" in cfg:  # GRU gates / state update and conv bias+ReLU as fused kernels (models/stereoanywhere/sa_kernels.cu)
        assert "hacc" in cfg and "gru" in cfg
        from .update import BasicMotionEncoder, UpdateHead
        for m in model.modules():
            if isinstance(m, ConvGRUType()):
                m._bzr32 = m._bzr.float().contiguous()
                m._bq32 = m.convq.bias.float().contiguous()
                m.forward = _gru_forward_k.__get__(m)
            elif isinstance(m, nn.Conv2d) and getattr(m, "_hacc", False) and m.bias is not None:
                m._b32 = m.bias.float().contiguous()
        for m in model.modules():
            if isinstance(m, BasicMotionEncoder):
                m._k = True
            elif isinstance(m, UpdateHead):
                m.forward = (lambda self, x: self.conv2(_conv_act(self.conv1, x))).__get__(m)
        mk = model.update_block.mask
        mk.forward = (lambda self, x: self[2](_conv_act(self[0], x))).__get__(mk)
        from . import update as upd
        upd.interp = _interp_keep  # GRU state interpolation in fp16 (autocast ran it in fp32 + cast back)
        ub = model.update_block
        if "gk2" in cfg and not ub.predict_confidence and ub.n_gru_layers == 3 and all(
                m.hidden_dim % 8 == 0 for m in (ub.gru08, ub.gru16, ub.gru32) if hasattr(m, "hidden_dim")):
            ub.forward = _update_forward_k.__get__(ub)
    if "ek" in cfg:  # encoders (fnet / cnet): conv -> norm -> ReLU (-> + residual -> ReLU) as conv + one fused pass
        from .extractor import ResidualBlock, BasicEncoder, MultiBasicEncoder
        for m in model.modules():
            if isinstance(m, ResidualBlock) and getattr(m.conv1, "_hacc", False):
                m.forward = _resblock_k.__get__(m)
            elif isinstance(m, (BasicEncoder, MultiBasicEncoder)):
                m.norm1._post_relu = True
                m.norm1.forward = _stem_norm_k.__get__(m.norm1)
                m.relu1 = nn.Identity()
    if "ek" in cfg and "c1f32" in cfg and isinstance(model.cnet.norm1, nn.BatchNorm2d) \
            and model.cnet.conv1.stride == (1, 1) and model.cnet.conv1.weight.shape[0] == 64 and model.cnet.conv1.kernel_size == (7, 7):
        # cnet stem (fp32 1-channel 7x7 conv + BN + ReLU) as one kernel writing fp16 NHWC
        c1, n1 = model.cnet.conv1, model.cnet.norm1
        c1._w49 = c1._w1.reshape(64, 49).contiguous()
        c1._s, c1._t = n1._s.float().contiguous(), n1._t.float().contiguous()
        c1.forward = _stem_k.__get__(c1)
        n1.forward = lambda x: x
        model.cnet.relu1 = nn.Identity()
    if "lookup" in cfg:
        from . import corr as corr_mod
        corr_mod.CorrBlock1D.__call__ = _lookup_k if "gk" in cfg else _lookup
        corr_mod.CorrBlock1D.__init__ = _lookup_init
    model._fast_cfg = set(cfg)
    if "vk" in cfg:
        from . import voltail
        from . import stereoanywhere as sam
        voltail.prepare(model)
        for i, n in enumerate(("estimate_left_disparity", "estimate_right_disparity")):
            setattr(sam, n, _est_wrap(getattr(sam, n), i))
        for i, n in enumerate(("estimate_left_confidence", "estimate_right_confidence")):
            setattr(sam, n, _est_wrap(getattr(sam, n), i))
    return n


def _est_wrap(f, i):
    if getattr(f, "_sa_wrapped", False):
        f = f._sa_orig

    def g(v, *a, **k):
        e = getattr(v, "_sa_est", None)
        if e is not None and not a and not k:
            return e[i]
        return f(v, *a, **k)
    g._sa_wrapped, g._sa_orig = True, f
    return g


def truncate_inplace(vol, disp, conf, att):
    """vol (1, 1, H, W, W3) fp32 *= truncate_corr_volume_v2(disp, conf, conf_th=None, attenuation_gain=att), one kernel."""
    from .kernels import Kernels
    _, _, H, W, D = vol.shape
    assert vol.is_contiguous() and vol.dtype == torch.float32
    disp = disp.float().contiguous()
    conf = conf.float().contiguous()
    k = Kernels.get()
    k.launch("trunc_mul", k.linear(H * W * D), vol, disp, conf, H, W, D, float(att))


def _fp32_wrap(fwd):
    def cast(v):
        if torch.is_tensor(v) and v.is_floating_point():
            return v.float()
        if isinstance(v, (list, tuple)):
            return type(v)(cast(t) for t in v)
        return v

    def f(*a, **k):
        with torch.autocast("cuda", enabled=False):
            return fwd(*cast(a), **{n: cast(v) for n, v in k.items()})
    return f


def _conv1_f32(self, x):
    with torch.autocast("cuda", enabled=False):
        y = F.conv2d(x[:, :1].float(), self._w1, self._b1, self.stride, self.padding)
    y = y.half()
    return y.contiguous(memory_format=torch.channels_last) if self._cl else y


def _stem_k(self, x):
    from .kernels import Kernels
    B, _, H, W = x.shape
    assert B == 1
    x0 = x[0, 0].float().contiguous()
    y = torch.empty((1, 64, H, W), device=x.device, dtype=torch.float16, memory_format=torch.channels_last)
    k = Kernels.get()
    k.launch("stem7_c64", (((W + 15) // 16, (H + 15) // 16, 1), (16, 16, 1)), x0, H, W, self._w49, self._b1, self._s, self._t, y)
    return y


def _bf16_wrap(fwd):
    def cast(v):
        if torch.is_tensor(v) and v.is_floating_point():
            return v.to(torch.bfloat16)
        if isinstance(v, (list, tuple)):
            return type(v)(cast(t) for t in v)
        return v

    def f(*a, **k):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fwd(*cast(a), **{n: cast(v) for n, v in k.items()})
    return f


def ConvGRUType():
    from .update import ConvGRU
    return ConvGRU


def _conv(m, x, w, b):
    if getattr(m, "_hacc", False):
        from .cudnn_conv import conv_half_acc
        return conv_half_acc(x.half(), w, b, list(m.padding), list(m.stride))
    return F.conv2d(x, w, b, m.stride, m.padding)


def _conv_hacc_forward(self, x):
    from .cudnn_conv import conv_half_acc
    return conv_half_acc(x.half(), self.weight, self.bias, list(self.padding), list(self.stride))


def _gru_forward(self, h, cz, cr, cq, *x_list):
    """ConvGRU with z|r convs merged into one conv and the [h | x] input assembled once in an fp16 NHWC buffer; r*h
    overwrites the h slice for the q conv (upstream: 2 cats, each re-laid-out / cast by the conv). Same math."""
    Hd = h.shape[1]
    cx = sum(t.shape[1] for t in x_list)
    hx = torch.empty((h.shape[0], Hd + cx, h.shape[2], h.shape[3]), device=h.device, dtype=torch.float16,
                     memory_format=torch.channels_last)
    hx[:, :Hd] = h
    c0 = Hd
    for t in x_list:
        hx[:, c0:c0 + t.shape[1]] = t
        c0 += t.shape[1]
    zr = _conv(self.convz, hx, self._wzr, self._bzr)
    z = torch.sigmoid(zr[:, :Hd] + cz)
    r = torch.sigmoid(zr[:, Hd:] + cr)
    hx[:, :Hd] = r * h
    q = torch.tanh(self.convq(hx) + cq)
    return ((1 - z) * h + z * q).to(torch.float16)


def _launch_rows(name, n):
    from .kernels import Kernels
    k = Kernels.get()
    return k, k.linear(n)


def _rows(t):
    """(N, C, H, W) channels-last-like view -> row stride (elements) or None."""
    if t.stride(1) != 1 or t.stride(2) != t.shape[3] * t.stride(3) or t.stride(0) != t.shape[2] * t.stride(2) and t.shape[0] > 1:
        return None
    return t.stride(3)


def _conv_act(m, x, act=1, out=None):
    """hacc conv + bias + ReLU with one epilogue pass (bias_act kernel). out: rows-layout view (N*H*W rows, row stride
    out.stride(3), batch stride = H*W*row stride or a channel offset for batch-as-channel-block views) to write into."""
    from .cudnn_conv import conv_half_acc
    y = conv_half_acc(x.half(), m.weight, None, list(m.padding), list(m.stride), out=out)
    C = y.shape[1]
    rs = y.stride(3)
    if y.shape[0] > 1 and y.stride(0) != y.shape[2] * y.shape[3] * rs:   # batch packed as channel blocks
        P, C2, rs2 = y.shape[2] * y.shape[3], y.shape[0] * C, rs
        assert y.stride(0) == C and rs2 >= C2
        k, cfg = _launch_rows("bias_act", P * C2 // 8)
        b = m._b32 if y.shape[0] == 1 else m._b32.repeat(y.shape[0])
        k.launch("bias_act", cfg, y, rs2, b, P, C2, act)
        return y
    P = y.shape[0] * y.shape[2] * y.shape[3]
    k, cfg = _launch_rows("bias_act", P * C // 8)
    k.launch("bias_act", cfg, y, rs, m._b32, P, C, act)
    return y


def _norm_st(norm, y, b=None):
    """per-(n, c) scale / shift (fp32 [N, C]) of norm(y + b) for instance norm (statistics of y; a conv bias cancels) or
    eval batch norm (folded, conv bias included)."""
    N, C = y.shape[:2]
    if isinstance(norm, (nn.InstanceNorm2d,)):
        var, mean = torch.var_mean(y, dim=(2, 3), unbiased=False)   # fp16 results, as the patched instance norm
        s = torch.rsqrt(var.float() + norm.eps)
        return s, (-mean.float() * s).contiguous()
    s, t = norm._s.float(), norm._t.float()
    if b is not None:
        t = t + b.float() * s
    return s.expand(N, C).contiguous(), t.expand(N, C).contiguous()


def _norm_act_launch(y, st, act1, r=None, rst=None, act2=0):
    N, C, H, W = y.shape
    assert C % 8 == 0 and y.is_contiguous(memory_format=torch.channels_last)
    from .kernels import Kernels
    k = Kernels.get()
    k.launch("norm_act", k.linear(N * H * W * C // 8), y, st[0], st[1], r if r is not None else 0,
             rst[0] if rst is not None else 0, rst[1] if rst is not None else 0, N, H * W, C, act1, act2)
    return y


def _conv_raw(m, x):
    from .cudnn_conv import conv_half_acc
    return conv_half_acc(x.half(), m.weight, None, list(m.padding), list(m.stride))


def _resblock_k(self, x):
    """extractor.ResidualBlock (instance or eval batch norm): relu(x' + relu(n2(c2(relu(n1(c1 x)))))), each conv followed
    by one fused normalize/act(/residual) pass; conv biases folded into the norm shift (cancel under instance norm)."""
    x = x.half().contiguous(memory_format=torch.channels_last)
    y = _conv_raw(self.conv1, x)
    _norm_act_launch(y, _norm_st(self.norm1, y, self.conv1.bias), 1)
    y = _conv_raw(self.conv2, y)
    st2 = _norm_st(self.norm2, y, self.conv2.bias)
    if self.downsample is not None:
        d = _conv_raw(self.downsample[0], x)
        return _norm_act_launch(y, st2, 1, d, _norm_st(self.norm3, d, self.downsample[0].bias), 1)
    return _norm_act_launch(y, st2, 1, x, None, 1)


def _stem_norm_k(self, x):
    """encoder stem norm1 + relu1 in one pass (x = conv1 output incl. bias)."""
    x = x.half().contiguous(memory_format=torch.channels_last)
    return _norm_act_launch(x, _norm_st(self, x), 1)


def _interp_keep(x, dest):
    """update.interp (bilinear, align_corners=True) in the input dtype: an fp16 upsample computes in fp32 and rounds once,
    = the autocast fp32 result rounded to fp16 by the consumer (the GRU input buffer is fp16)."""
    with torch.autocast("cuda", enabled=False):
        return F.interpolate(x, dest.shape[2:], mode="bilinear", align_corners=True)


def _gru_prep(h, cz, cr, cq):
    Hd = h.shape[1]
    if h.dtype != torch.float16 or _rows(h) != Hd:
        h = h.half().contiguous(memory_format=torch.channels_last)
    cs = _rows(cz)
    if cs is None or _rows(cr) != cs or _rows(cq) != cs:
        cz, cr, cq = (t.half().contiguous(memory_format=torch.channels_last) for t in (cz, cr, cq))
        cs = Hd
    return h, cz, cr, cq, cs


def _gru_core(self, h, cz, cr, cq, cs, hx):
    """hx = [h | x] (fp16 channels-last) already assembled -> new hidden state."""
    from .cudnn_conv import conv_half_acc
    N, Hd, H, W = h.shape
    zr = conv_half_acc(hx, self._wzr, None, list(self.convz.padding), list(self.convz.stride))
    P = N * H * W
    k, cfg = _launch_rows("gru_zr", P * Hd // 8)
    k.launch("gru_zr", cfg, zr, 2 * Hd, self._bzr32, cz, cr, cs, h, Hd, hx, hx.shape[1], P, Hd)
    q = conv_half_acc(hx, self.convq.weight, None, list(self.convq.padding), list(self.convq.stride))
    out = torch.empty_like(h)
    k.launch("gru_out", cfg, q, Hd, self._bq32, cq, cs, zr, 2 * Hd, h, Hd, out, Hd, P, Hd)
    return out


def _gru_forward_k(self, h, cz, cr, cq, *x_list):
    """_gru_forward with the gates as two fused kernels: gru_zr (bias, + cz / cr, sigmoids, r * h into the conv input
    buffer, z kept in the zr buffer) and gru_out (bias, + cq, tanh, (1 - z) h + z q). Upstream fp16 rounding steps kept."""
    h, cz, cr, cq, cs = _gru_prep(h, cz, cr, cq)
    Hd = h.shape[1]
    cx = sum(t.shape[1] for t in x_list)
    N, _, H, W = h.shape
    hx = torch.empty((N, Hd + cx, H, W), device=h.device, dtype=torch.float16, memory_format=torch.channels_last)
    hx[:, :Hd] = h
    c0 = Hd
    for t in x_list:
        hx[:, c0:c0 + t.shape[1]] = t
        c0 += t.shape[1]
    return _gru_core(self, h, cz, cr, cq, cs, hx)


def _motion_into(enc, flow, corr, corr_mono, dst):
    """BasicMotionEncoder writing its output [relu(_conv(...)) | flow] into dst (a channel slice of the GRU input buffer)."""
    n, _, H, W = corr.shape
    assert n == 1
    c = torch.cat([corr, corr_mono], 0)
    c = _conv_act(enc.convc1, c)
    # strided cuDNN outputs: fine for a channel slice of a 1x64-ch conv (convf2), but the batch-as-channel-block view for
    # convc2 and the 384-ch row slice for _conv picked much slower engines (Midd-H +170 / +45 ms) -> copies there
    cf = torch.empty((1, 192, H, W), device=c.device, dtype=torch.float16, memory_format=torch.channels_last)
    c = _conv_act(enc.convc2, c)
    cf[:, :64] = c[:1]
    cf[:, 64:128] = c[1:]
    _conv_act(enc.convf2, _conv_act(enc.convf1, flow), out=cf[:, 128:])
    co = enc._conv.weight.shape[0]
    dst[:, :co] = _conv_act(enc._conv, cf)
    dst[:, co:] = flow


def _resample_into(name, src, dst):
    """bilerp_rows / pool2x_rows from an NHWC fp16 batch-1 tensor into dst (channel slice of a channels-last buffer)."""
    if src.dtype != torch.float16 or _rows(src) != src.shape[1]:
        src = src.half().contiguous(memory_format=torch.channels_last)
    _, C, Hs, Ws = src.shape
    _, _, Hd, Wd = dst.shape
    from .kernels import Kernels
    k = Kernels.get()
    k.launch(name, k.linear(Hd * Wd * C // 8), src, C, Hs, Ws, dst, dst.stride(3), Hd, Wd, C)


def _update_forward_k(self, net, inp, corr=None, corr_mono=None, flow=None, flow_conf=None, iter08=True, iter16=True,
                      iter32=True, update=True, upsample=True):
    """BasicMultiUpdateBlock.forward (3 GRU levels, no confidence head, batch 1): every GRU input [h | x...] is written in
    place by its producers (pool2x / bilinear interp kernels, motion encoder)."""
    def gru(g, h, cz, cr, cq, xs):
        h, cz, cr, cq, cs = _gru_prep(h, cz, cr, cq)
        N, Hd, H, W = h.shape
        hx = torch.empty((N, Hd + sum(c for c, _ in xs), H, W), device=h.device, dtype=torch.float16,
                         memory_format=torch.channels_last)
        hx[:, :Hd] = h
        c0 = Hd
        for c, fill in xs:
            fill(hx[:, c0:c0 + c], h)
            c0 += c
        return _gru_core(g, h, cz, cr, cq, cs, hx)
    pool = lambda src: (src.shape[1], lambda dst, h: _resample_into("pool2x_rows", src, dst))
    up = lambda src: (src.shape[1], lambda dst, h: _resample_into("bilerp_rows", src, dst))
    if iter32:
        net[2] = gru(self.gru32, net[2], *inp[2], [pool(net[1])])
    if iter16:
        net[1] = gru(self.gru16, net[1], *inp[1], [pool(net[0]), up(net[2])])
    if iter08:
        cm = self.encoder._conv.weight.shape[0] + 2
        mot = (cm, lambda dst, h: _motion_into(self.encoder, flow, corr, corr_mono, dst))
        net[0] = gru(self.gru08, net[0], *inp[0], [mot, up(net[1])])
    if not update:
        return net
    delta_flow = self.flow_head(net[0])
    mask = .25 * self.mask(net[0]) if upsample else None
    return net, mask, delta_flow


def _motion_forward(self, flow, corr, corr_mono):
    """BasicMotionEncoder with the stereo/mono correlation branches batched through the shared convc1/convc2 and an
    fp16 output (upstream cat with the fp32 flow promoted the whole GRU input to fp32 -> casts + copies per conv)."""
    n = corr.shape[0]
    c = torch.cat([corr, corr_mono], 0)
    if getattr(self, "_k", False):
        c = _conv_act(self.convc2, _conv_act(self.convc1, c))
        flo = _conv_act(self.convf2, _conv_act(self.convf1, flow))
        out = _conv_act(self._conv, torch.cat([c[:n], c[n:], flo], dim=1))
        return torch.cat([out, flow.to(out.dtype)], dim=1)
    c = F.relu(self.convc1(c), inplace=True)
    c = F.relu(self.convc2(c), inplace=True)
    flo = F.relu(self.convf1(flow), inplace=True)
    flo = F.relu(self.convf2(flo), inplace=True)
    out = F.relu(self._conv(torch.cat([c[:n], c[n:], flo], dim=1)), inplace=True)
    return torch.cat([out, flow.to(out.dtype)], dim=1)


def _lookup_init(self, fullcorr, num_levels=4, radius=4, pad=[0, 0]):
    """1D all-pairs pyramid; level 0 is a view of the volume (no copy)."""
    self.num_levels, self.radius, self.pad = num_levels, radius, pad
    b, h1, w1, dim, w2 = fullcorr.shape
    corr = fullcorr.reshape(b * h1 * w1, 1, 1, w2).float()
    self.corr_pyramid = []
    for i in range(num_levels):
        self.corr_pyramid.append(corr.reshape(b * h1 * w1, -1))
        if i + 1 < num_levels:
            corr = F.avg_pool2d(corr, [1, 2], stride=[1, 2])


def _lookup_k(self, coords):
    """_lookup as one kernel (all levels, fp16 channels-last output (B, 4(2r+1), H, W))."""
    assert len(self.corr_pyramid) == 4 and self.pad[0] == 0 and self.pad[1] == 0
    b, _, h1, w1 = coords.shape
    x = coords[:, 0]
    if x.dtype != torch.float32:
        x = x.float()
    K = 2 * self.radius + 1
    out = torch.empty((b, h1, w1, 4 * K), device=coords.device, dtype=torch.float16)
    from .kernels import Kernels
    k = Kernels.get()
    c = self.corr_pyramid
    assert all(t.is_contiguous() for t in c)
    k.launch("corr_lookup", k.linear(b * h1 * w1), c[0], c[1], c[2], c[3], c[0].shape[1], c[1].shape[1], c[2].shape[1],
             c[3].shape[1], x, x.stride(0), x.stride(1), x.stride(2), b, h1, w1, out, self.radius)
    return out.permute(0, 3, 1, 2)


def _lookup(self, coords):
    """== CorrBlock1D.__call__ (grid_sample, align_corners=True, zero padding) as a gather + lerp with out-of-range taps
    zeroed; no host sync (upstream asserts torch.unique(ygrid) on every call)."""
    r = self.radius
    x = coords[:, :1].permute(0, 2, 3, 1) + self.pad[0]  # B H W 1
    b, h1, w1, _ = x.shape
    dx = torch.arange(-r, r + 1, device=coords.device, dtype=torch.float32)
    out = []
    for i, corr in enumerate(self.corr_pyramid):
        wl = corr.shape[1]
        xs = x.reshape(-1, 1).float() / 2 ** i + dx  # BHW 2r+1
        x0 = torch.floor(xs)
        a = xs - x0
        i0 = x0.long()
        i1 = i0 + 1
        v0 = torch.gather(corr, 1, i0.clamp(0, wl - 1)) * ((i0 >= 0) & (i0 < wl))
        v1 = torch.gather(corr, 1, i1.clamp(0, wl - 1)) * ((i1 >= 0) & (i1 < wl))
        v = v0 * (1 - a) + v1 * a
        out.append(v.view(b, h1, w1, -1)[:, :, self.pad[0]:w1 - self.pad[1]])
    return torch.cat(out, dim=-1).permute(0, 3, 1, 2).contiguous().to(coords.dtype)


def forward_fast(model, L, R, ml, mr, iters=32):
    if "cl" in model._fast_cfg:
        L, R, ml, mr = (t.contiguous(memory_format=torch.channels_last) for t in (L, R, ml, mr))
    with torch.autocast("cuda", dtype=torch.float16):
        d, _ = model(L, R, ml, mr, test_mode=True, iters=iters)
    return d


# ------------------------------------------------------------------------------------------- mono volume (fp16)
def _att(m, cv, fl, fr):
    """DoubleFeatureAtt: cv * interp(sigmoid(att_l) * sigmoid(att_r)). When the attention maps already have the volume
    size (inputs padded to /32: always), the interpolation is the identity -> broadcast multiply in place, no
    full-size attention volume."""
    la = torch.sigmoid(m.feat_att_left(fl)).unsqueeze(2)                     # B C 1 H W
    ra = torch.sigmoid(m.feat_att_right(fr)).permute(0, 1, 3, 2).unsqueeze(4)  # B C W' H 1
    if la.shape[3:] == cv.shape[3:] and ra.shape[2:4] == cv.shape[2:4]:
        return cv.mul_(la).mul_(ra)
    _cv = F.interpolate(la * ra, size=cv.shape[2:], mode="trilinear", align_corners=True)
    return cv * _cv


def _interp3(x, shape):
    return F.interpolate(x, size=tuple(shape[2:]), mode="trilinear", align_corners=True)


def _split_1x1(bc, a, b, b_up_shape=None, a_up_shape=None):
    """BasicConv(1x1x1, no bias) on cat([a, b]) without the cat: conv_a(a) + conv_b(b); a 1x1 conv commutes with
    trilinear interpolation, so an input that upstream upsamples first is convolved at low resolution and upsampled
    with fewer channels (exact up to rounding)."""
    w = bc.conv.weight
    ca = a.shape[1]
    ya = F.conv3d(a, w[:, :ca])
    yb = F.conv3d(b, w[:, ca:])
    if a_up_shape is not None:
        ya = _interp3(ya, a_up_shape)
    if b_up_shape is not None:
        yb = _interp3(yb, b_up_shape)
    return bc.act_fn(bc.norm_fn(ya.add_(yb)))


def hourglass_fast(hg, x, fl, fr):
    """Hourglass.forward on the volume already in its internal layout (B C W3 H W2), fp16, returns internal layout."""
    ns = hg.number_of_scales
    orig = x
    downs = []
    for i in range(ns - 1):
        x = hg.down_layers[i](x)
        x = _att(hg.feature_atts[i], x, fl[i + 1], fr[i + 1])
        downs.append(x)
    for i in range(ns - 2):
        low, skip = downs[ns - 2 - i], downs[ns - 3 - i]
        seq = hg.agg_layers[i]
        y = _split_1x1(seq[0], low, skip, a_up_shape=skip.shape)
        y = seq[2](seq[1](y))
        x = _att(hg.feature_atts_up[i], y, fl[ns - 2 - i], fr[ns - 2 - i])
    del downs, low, skip
    y = _split_1x1(hg.final_agg[0], orig, x, b_up_shape=orig.shape)
    del x
    y = hg.final_agg[2](hg.final_agg[1](y))
    return _att(hg.final_feature_atts_up, y, fl[0], fr[0])


STREAM_BYTES = 2 * 2**30   # masked mono volume larger than this (fp16) -> streamed hourglass (exact, less memory)
STREAM_CHUNK = 48          # slices along the first volume axis per chunk


def hourglass_stream(hg, make, shape, fl, fr, wcls, chunk=STREAM_CHUNK):
    """hourglass_fast without ever materializing the full-resolution masked input volume: it is rebuilt slice-chunk by
    slice-chunk (cheap: volume x masks) for the two layers that read it, the first stride-2 3x3x3 conv (chunks with a
    1-slice halo, zero padding as upstream) and the final 1x1x1 conv on cat([orig, up(x)]) (pointwise). The upsampling
    of the low-resolution branch is trilinear = linear along the last two axes (one interpolate, size along axis 2
    unchanged = identity) then linear along axis 2 per chunk with the same align_corners=True weights. Same math as
    hourglass_fast up to rounding."""
    B, C, D, H, W = shape
    ns = hg.number_of_scales
    bc0 = hg.down_layers[0][0]
    cv = bc0.conv
    Do = (D + 2 * cv.padding[0] - 3) // cv.stride[0] + 1
    out = None
    for oa in range(0, Do, chunk):
        ob = min(Do, oa + chunk)
        ia, ie = 2 * oa - 1, 2 * ob          # input slices ia .. ie (inclusive) feed outputs oa .. ob-1
        xa, xb = max(0, ia), min(D, ie + 1)
        xc = F.pad(make(xa, xb), (0, 0, 0, 0, xa - ia, ie + 1 - xb))
        yc = F.conv3d(xc, cv.weight, cv.bias, cv.stride, (0, cv.padding[1], cv.padding[2]))
        del xc
        if out is None:
            out = torch.empty((B, yc.shape[1], Do, yc.shape[3], yc.shape[4]), device=yc.device, dtype=yc.dtype)
        out[:, :, oa:ob] = yc
    x = bc0.act_fn(bc0.norm_fn(out))
    del out
    x = hg.down_layers[0][1](x)
    x = _att(hg.feature_atts[0], x, fl[1], fr[1])
    downs = [x]
    for i in range(1, ns - 1):
        x = hg.down_layers[i](x)
        x = _att(hg.feature_atts[i], x, fl[i + 1], fr[i + 1])
        downs.append(x)
    for i in range(ns - 2):
        low, skip = downs[ns - 2 - i], downs[ns - 3 - i]
        seq = hg.agg_layers[i]
        y = _split_1x1(seq[0], low, skip, a_up_shape=skip.shape)
        y = seq[2](seq[1](y))
        x = _att(hg.feature_atts_up[i], y, fl[ns - 2 - i], fr[ns - 2 - i])
    del downs, low, skip
    bc0, bc1, bc2 = hg.final_agg
    w0 = bc0.conv.weight
    yb = F.conv3d(x, w0[:, C:])
    del x
    d = yb.shape[2]
    yb = F.interpolate(yb, size=(d, H, W), mode="trilinear", align_corners=True)
    sc = (d - 1) / (D - 1) if D > 1 else 0.0
    dt = yb.dtype

    # The full-resolution tail (final_agg: 1x1 conv + 2 x 3x3x3 conv, each with instance norm, then the feature
    # attention and the classifier conv) is evaluated in slice chunks with recomputation: instance-norm statistics are
    # global, so each norm gets its own pass (chunk statistics merged exactly, Chan et al.), and every pass recomputes
    # the chunk from the masked volume with the 3x3x3 halo it needs. Only the 2-channel classifier output is stored.
    def y0(a, b):
        src = torch.arange(a, b, device=yb.device, dtype=torch.float32) * sc
        j0 = src.floor().long().clamp_(0, d - 1)
        j1 = (j0 + 1).clamp_(max=d - 1)
        t = (src - j0.float()).view(1, 1, -1, 1, 1)
        up = yb.index_select(2, j0).float().mul_(1 - t).add_(yb.index_select(2, j1).float().mul_(t))
        return F.conv3d(make(a, b), w0[:, :C]).float().add_(up).to(dt)

    def stats(f, eps):
        n, mean, m2 = 0, None, None
        for a in range(0, D, chunk):
            b = min(D, a + chunk)
            xc = f(a, b).float()
            v, m = torch.var_mean(xc, dim=(2, 3, 4), keepdim=True, unbiased=False)
            k = xc[0, 0].numel()
            if mean is None:
                n, mean, m2 = k, m, v * k
            else:
                delta = m - mean
                tot = n + k
                mean = mean + delta * (k / tot)
                m2 = m2 + v * k + delta * delta * (n * k / tot)
                n = tot
        return mean.to(dt), torch.rsqrt(m2 / n + eps).to(dt)

    def norm_act(bc, st):
        mean, rstd = st
        return lambda f: (lambda a, b: bc.act_fn(f(a, b).sub_(mean).mul_(rstd)))

    def conv_halo(wt, f):
        def g(a, b):
            xa, xb = max(0, a - 1), min(D, b + 1)
            xc = F.pad(f(xa, xb), (0, 0, 0, 0, xa - (a - 1), (b + 1) - xb))
            return F.conv3d(xc, wt, None, 1, (0, 1, 1))
        return g

    a0 = norm_act(bc0, stats(y0, bc0.norm_fn.eps))(y0)
    z1 = conv_halo(bc1.conv.weight, a0)
    a1 = norm_act(bc1, stats(z1, bc1.norm_fn.eps))(z1)
    z2 = conv_halo(bc2.conv.weight, a1)
    a2n = norm_act(bc2, stats(z2, bc2.norm_fn.eps))(z2)
    att = hg.final_feature_atts_up
    la = torch.sigmoid(att.feat_att_left(fl[0])).unsqueeze(2)
    ra = torch.sigmoid(att.feat_att_right(fr[0])).permute(0, 1, 3, 2).unsqueeze(4)
    assert la.shape[3:] == (H, W) and ra.shape[2:4] == (D, H)
    a2 = lambda a, b: a2n(a, b).mul_(la).mul_(ra[:, :, a:b])
    cls = conv_halo(wcls, a2)
    out = torch.empty((B, wcls.shape[0], D, H, W), device=yb.device, dtype=dt)
    for a in range(0, D, chunk):
        b = min(D, a + chunk)
        out[:, :, a:b] = cls(a, b)
    return out


def mono_volume_fast(model, vol, lm, rm, fmde2, fmde3):
    """masked mono volume -> hourglass -> classifiers, all fp16, no layout copies. vol [B 1 H W2 W3], masks [B N H W]."""
    assert model.args.n_additional_hourglass == 0
    if "vk" in model._fast_cfg and vol.shape[0] == 1 and lm.shape[1] == 8:
        from .voltail import mono_volume_vk
        with torch.autocast("cuda", enabled=False):
            return mono_volume_vk(model, vol, lm, rm, fmde2, fmde3, _att, _interp3, _split_1x1)
    with torch.autocast("cuda", enabled=False):
        B, _, H, W2, W3 = vol.shape
        N = lm.shape[1]
        # internal hourglass layout (B C W3 H W2) = masked.permute(0, 1, 4, 2, 3)
        dt = torch.float32 if "hg32" in model._fast_cfg else torch.float16
        fl = [f.to(dt) for f in fmde2]
        fr = [f.to(dt) for f in fmde3]
        vp = vol.permute(0, 1, 4, 2, 3)                 # B 1 W3 H W2 (view)
        lmd = lm.to(dt).unsqueeze(2)                    # B N 1 H W2
        rmp = rm.to(dt).permute(0, 1, 3, 2).unsqueeze(4)  # B N W3 H 1
        stream = "stream" in model._fast_cfg or B * N * W3 * H * W2 * 2 > STREAM_BYTES
        # classifiers on the internal layout with permuted kernels; output permuted back (= upstream layout)
        # both classifiers (same input, no bias) as one conv with 2 output channels
        w = torch.cat([model.classifier_mono.weight, model.classifier_monoconf.weight], 0).permute(0, 1, 4, 2, 3).contiguous()
        if stream:
            def make(a, b):
                return (vp[:, :, a:b].to(dt) * lmd).mul_(rmp[:, :, a:b])
            o = hourglass_stream(model.hourglass_mono, make, (B, N, W3, H, W2), fl, fr, w)
        else:
            x = torch.empty((B, N, W3, H, W2), device=vol.device, dtype=dt)
            torch.mul(vp.to(dt), lmd, out=x)
            x.mul_(rmp)
            y = hourglass_fast(model.hourglass_mono, x, fl, fr)
            del x
            o = F.conv3d(y, w, None, 1, 1)
        o = o.permute(0, 1, 3, 4, 2)
    return o[:, :1].float(), o[:, 1:].float()  # separate storages: the confidence volume is freed early upstream


def _fast_interpolate(orig):
    import numpy as np

    def f(input, size=None, scale_factor=None, mode="nearest", align_corners=None, recompute_scale_factor=None,
          antialias=False):
        x = input
        if (mode != "bilinear" or antialias or recompute_scale_factor or x.dim() != 4 or x.dtype != torch.float16
                or x.shape[1] % 8 or not x.is_contiguous(memory_format=torch.channels_last) or x.requires_grad):
            return orig(input, size, scale_factor, mode, align_corners, recompute_scale_factor, antialias)
        N, C, Hs, Ws = x.shape
        if size is not None:
            Hd, Wd = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
            sf = None
        else:
            sf = scale_factor if isinstance(scale_factor, (tuple, list)) else (scale_factor, scale_factor)
            Hd, Wd = int(np.floor(Hs * sf[0])), int(np.floor(Ws * sf[1]))
        if align_corners:
            sh = np.float32((Hs - 1) / (Hd - 1)) if Hd > 1 else np.float32(0)
            sw = np.float32((Ws - 1) / (Wd - 1)) if Wd > 1 else np.float32(0)
        elif sf is not None:
            sh, sw = np.float32(1.0 / sf[0]), np.float32(1.0 / sf[1])
        else:
            sh, sw = np.float32(Hs / Hd), np.float32(Ws / Wd)
        y = torch.empty((N, C, Hd, Wd), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        from .kernels import Kernels
        k = Kernels.get()
        k.launch("bilerp_nhwc", k.linear(N * Hd * Wd * C // 8), x, N, Hs, Ws, y, Hd, Wd, C, float(sh), float(sw),
                 1 if align_corners else 0)
        return y
    return f


def _interp_scope(fwd):
    """run fwd with torch.nn.functional.interpolate routed through bilerp_nhwc where it applies."""
    def g(*a, **k):
        orig = F.interpolate
        F.interpolate = _fast_interpolate(orig)
        try:
            return fwd(*a, **k)
        finally:
            F.interpolate = orig
    return g


def _gacc_wrap(fwd):
    def f(*a, **k):
        old = torch.backends.cuda.matmul.allow_fp16_accumulation
        torch.backends.cuda.matmul.allow_fp16_accumulation = True
        try:
            return fwd(*a, **k)
        finally:
            torch.backends.cuda.matmul.allow_fp16_accumulation = old
    return f


def _block_forward(self, x):
    """dinov2 Block (eval, no drop path): x + ls1(attn(norm1 x)) + ls2(mlp(norm2 .)) as addcmuls; the block input is not
    modified (get_intermediate_layers keeps references to block outputs)."""
    x = torch.addcmul(x, self.attn(self.norm1(x)), self.ls1.gamma)
    return x.addcmul_(self.mlp(self.norm2(x)), self.ls2.gamma)


@torch.no_grad()
def prepare_mono(mono, cfg=()):
    """Depth Anything V2: pure fp16 weights once. Opt-in "mhacc": DPT-head convs as fp16-accumulate cuDNN convs (NHWC);
    measured worse on ETH3D (D1 1.90 vs 1.69 without), so not used by default."""
    mono.half().eval()
    n = 0
    if "mcl" in cfg:  # DPT head convs with channels-last weights (activations are channels-last there: NCHW weights
        # made cuDNN transpose per call / pick slow kernels); fp32 accumulation unchanged
        for m in mono.depth_head.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
    if "mup" in cfg:  # DPT head bilinear resizes (fp16 NHWC) as one vectorized kernel each (torch's NHWC kernel ~5x slower)
        mono.depth_head.forward = _interp_scope(mono.depth_head.forward)
    if "mgacc" in cfg:  # ViT linears as fp16-accumulate cuBLAS GEMMs (flag set only while the ViT runs)
        mono.pretrained.get_intermediate_layers = _gacc_wrap(mono.pretrained.get_intermediate_layers)
    if "mres" in cfg:  # residual + LayerScale as one in-place addcmul per branch (one rounding instead of two)
        from ..depth_anything_v2.dinov2_layers.block import Block
        for m in mono.modules():
            if isinstance(m, Block):
                m.forward = _block_forward.__get__(m)
    if "mhacc" in cfg:
        for m in mono.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1 and m.dilation == (1, 1) and m.padding_mode == "zeros":
                m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last)
                m.forward = _conv_hacc_forward.__get__(m)
                m._hacc = True
                n += 1
    _fastbb().install_vit(mono.pretrained, __import__("os").environ.get("FASTBB_VIT", "fh-g1"))  # fastbb ViT path (FASTBB_VIT env: spec, "off" = previous path)
    return n


def _fastbb():
    """shared fast-backbone package (pip install git+https://github.com/fastforks/fastbb)."""
    import fastbb
    return fastbb
