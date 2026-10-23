"""Mono-volume hourglass with custom kernels for the full-resolution (8-channel) part (option "vk").

Layout: volumes NDHWC fp16 (internal hourglass axes D = W3, H, W = W2), batch 1. The masked input volume is never
needed by the tail: mvol_y0 rebuilds it on the fly from the mono correlation volume and the masks, fuses the split
1x1 conv + trilinear upsampling of the low-resolution branch and accumulates the instance-norm statistics; the two
3x3x3 convs and the classifier (conv3_c8, tensor cores) apply the previous instance norm + LeakyReLU (+ the final
feature attention) while loading their input and accumulate their own output statistics. Statistics in double, no
host syncs. Lower hourglass levels: cuDNN on channels_last_3d. Large volumes: same kernels in D-chunks with
recomputation (exact: global statistics are complete before each norm is applied).
"""
import torch
import torch.nn.functional as F

from .kernels import Kernels

STREAM_BYTES = 2 * 2**30   # 8-ch fp16 volume larger than this -> chunked tail (no full-resolution volume stored)
CHUNK = 32                 # D planes per chunk (stream mode)


def _k():
    return Kernels.get()


def pack_w3(w):
    """[Co<=8, 8, 3, 3, 3] -> mma B fragments [14][2][32] (half2 as int32): k = tap * 8 + ci, tap = kd * 9 + kh * 3 + kw."""
    co = w.shape[0]
    wk = torch.zeros((8, 28, 8), dtype=torch.float16, device=w.device)       # n, tap (27 + 1 zero), ci
    wk[:co, :27] = w.detach().half().reshape(co, 8, 27).permute(0, 2, 1)
    lane = torch.arange(32, device=w.device)
    g, t = lane // 4, lane % 4
    out = torch.empty((14, 2, 32, 2), dtype=torch.float16, device=w.device)
    for j in range(14):
        for hk in range(2):
            tap = 2 * j + hk
            out[j, hk, :, 0] = wk[g, tap, 2 * t]
            out[j, hk, :, 1] = wk[g, tap, 2 * t + 1]
    return out.view(torch.int32).reshape(14, 2, 32).contiguous()


def mvol(vol, lm, rm, dims, a, b, mode=0, w0=None, yb=None, out=None, out_d0=0, st=None):
    D, H, W = dims
    if b <= a:
        return
    Di = Hi = Wi = 0
    if yb is not None:
        Di, Hi, Wi = yb.shape[0], yb.shape[1], yb.shape[2]
    grid = ((b - a + 31) // 32, (W + 31) // 32, H)
    _k().launch("mvol_y0", (grid, (32, 8, 1)), vol, lm, rm, D, H, W, a, b, mode, w0 if w0 is not None else 0,
                yb if yb is not None else 0, Di, Hi, Wi, out if out is not None else 0, out_d0, st if st is not None else 0)


def conv3(x, x_d0, dims, oa, ob, st_in, eps, wp, y=None, y_d0=0, co_store=8, st_out=None, la=None, ra=None):
    D, H, W = dims
    if ob <= oa:
        return
    grid = ((W + 31) // 32, (H + 7) // 8, (ob - oa + 5) // 6)
    _k().launch("conv3_c8", (grid, (256, 1, 1)), x, x_d0, x.shape[0], D, H, W, oa, ob, st_in, 1.0 / (D * H * W), eps,
                la if la is not None else 0, ra if ra is not None else 0, wp, y if y is not None else 0, y_d0, co_store,
                st_out if st_out is not None else 0)


def _cl(t):
    """[D, H, W, C] buffer -> (1, C, D, H, W) channels_last_3d view (canonical batch stride)."""
    return t.unsqueeze(0).permute(0, 4, 1, 2, 3)


def _canon(x):
    if x.shape[0] == 1 and x.dim() == 5 and x.is_contiguous(memory_format=torch.channels_last_3d) and x.stride(0) != x.numel():
        st = list(x.stride()); st[0] = x.numel()
        x = x.as_strided(x.shape, st)
    return x


def _att_maps(m, fl, fr, shape):
    """DoubleFeatureAtt maps for a (1, C, D, H, W) volume: la [H][W][C], ra [D][H][C] (fp16), or None when the maps do not
    have the volume size (then the generic broadcast path is used)."""
    la = torch.sigmoid(m.feat_att_left(fl))      # 1 C H W
    ra = torch.sigmoid(m.feat_att_right(fr))     # 1 C H D
    C, D, H, W = shape[1:]
    if tuple(la.shape[1:]) != (C, H, W) or tuple(ra.shape[1:]) != (C, H, D):
        return None
    return la[0].permute(1, 2, 0).contiguous().half(), ra[0].permute(2, 1, 0).contiguous().half()


def norm_act3(bc, y, att=None):
    """BasicConv instance norm + LeakyReLU (+ feature attention) on a conv output, one fused pass (stats via var_mean)."""
    y = _canon(y.contiguous(memory_format=torch.channels_last_3d))
    C = y.shape[1]
    var, mean = torch.var_mean(y, dim=(2, 3, 4), unbiased=False)
    s = torch.rsqrt(var.float() + bc.norm_fn.eps)
    t = -mean.float() * s
    D, H, W = y.shape[2:]
    k = _k()
    k.launch("norm_lrelu_att", k.linear(D * H * W * C // 8), y, s, t, att[0] if att else 0, att[1] if att else 0,
             D, H, W, C)
    return y


def bconv(bc, x, att=None):
    return norm_act3(bc, bc.conv(_canon(x)), att)


def prepare(model):
    hg = model.hourglass_mono
    for m in hg.modules():
        if isinstance(m, torch.nn.Conv3d):
            m.weight.data = m.weight.data.contiguous(memory_format=torch.channels_last_3d)
    fa = hg.final_agg
    C = hg.down_layers[0][0].conv.weight.shape[1]
    assert fa[1].conv.weight.shape[:2] == (8, 8) and fa[2].conv.weight.shape[:2] == (8, 8)
    wcls = torch.cat([model.classifier_mono.weight, model.classifier_monoconf.weight], 0).permute(0, 1, 4, 2, 3)
    model._vk = dict(
        C=C, w0a=fa[0].conv.weight[:, :C, 0, 0, 0].float().contiguous(),
        w0b=fa[0].conv.weight[:, C:].contiguous(memory_format=torch.channels_last_3d),
        wp1=pack_w3(fa[1].conv.weight), wp2=pack_w3(fa[2].conv.weight), wpc=pack_w3(wcls),
        eps=(fa[0].norm_fn.eps, fa[1].norm_fn.eps, fa[2].norm_fn.eps))


def mono_volume_vk(model, vol, lm, rm, fmde2, fmde3, att, interp3, split_1x1, chunk=CHUNK):
    """== fast.mono_volume_fast (fp16) for batch 1; returns (disp volume, conf volume) in the upstream layout."""
    from .fast import _att
    hg, P = model.hourglass_mono, model._vk
    B, _, H, W2, W3 = vol.shape
    assert B == 1 and lm.shape[1] == 8
    D, W = W3, W2
    dims = (D, H, W)
    dev = vol.device
    v = vol[0, 0].float().contiguous()                     # [H][W2][W3]
    lm32 = lm[0].float().contiguous()                      # [8][H][W2]
    rm32 = rm[0].float().contiguous()                      # [8][H][W3]
    fl = [f.half() for f in fmde2]
    fr = [f.half() for f in fmde3]
    stream = "stream" in model._fast_cfg or D * H * W * 16 > STREAM_BYTES

    # ---- first stride-2 conv on the masked volume (cuDNN, NDHWC), in chunks when streaming
    bc0 = hg.down_layers[0][0]
    cv = bc0.conv
    Do = (D + 2 - 3) // 2 + 1
    if not stream:
        M = torch.empty((D, H, W, 8), device=dev, dtype=torch.float16)
        mvol(v, lm32, rm32, dims, 0, D, 0, out=M)
        x = F.conv3d(_cl(M), cv.weight, None, 2, 1)
        del M
    else:
        x = None
        for oa in range(0, Do, chunk):
            ob = min(Do, oa + chunk)
            ia, ie = 2 * oa - 1, 2 * ob                    # input planes ia .. ie (inclusive)
            buf = torch.zeros((ie + 1 - ia, H, W, 8), device=dev, dtype=torch.float16)
            mvol(v, lm32, rm32, dims, max(0, ia), min(D, ie + 1), 0, out=buf, out_d0=ia)
            yc = F.conv3d(_cl(buf), cv.weight, None, 2, (0, 1, 1))
            del buf
            if x is None:
                x = torch.empty((1, yc.shape[1], Do, yc.shape[3], yc.shape[4]), device=dev, dtype=yc.dtype,
                                memory_format=torch.channels_last_3d)
            x[:, :, oa:ob] = yc
    if "vk1" in model._fast_cfg:  # debug: lower levels with the generic modules
        x = bc0.act_fn(bc0.norm_fn(x))
        x = hg.down_layers[0][1](_canon(x))
        x = att(hg.feature_atts[0], x, fl[1], fr[1])
        ns = hg.number_of_scales
        downs = [x]
        for i in range(1, ns - 1):
            x = hg.down_layers[i](_canon(x))
            x = att(hg.feature_atts[i], x, fl[i + 1], fr[i + 1])
            downs.append(x)
        for i in range(ns - 2):
            low, skip = downs[ns - 2 - i], downs[ns - 3 - i]
            seq = hg.agg_layers[i]
            y = split_1x1(seq[0], _canon(low), _canon(skip), a_up_shape=skip.shape)
            y = seq[2](_canon(seq[1](_canon(y))))
            x = att(hg.feature_atts_up[i], y, fl[ns - 2 - i], fr[ns - 2 - i])
        del downs, low, skip
    else:
        # lower levels: cuDNN convs (NDHWC) + one fused norm / LeakyReLU / attention pass per BasicConv
        def att_or_generic(bc, y, m, a, b):
            maps = _att_maps(m, a, b, y.shape)
            if maps is None:
                return att(m, norm_act3(bc, y), a, b)
            return norm_act3(bc, y, maps)
        x = norm_act3(bc0, x)
        x = att_or_generic(hg.down_layers[0][1], hg.down_layers[0][1].conv(_canon(x)), hg.feature_atts[0], fl[1], fr[1])
        ns = hg.number_of_scales
        assert ns >= 3
        # upstream Hourglass: every aggregation step i upsamples downsampled_features[ns-2-i] (not the previous step's
        # output), and only the last step's output (i = ns-3: downs[1] up + downs[0]) reaches final_agg -> the deeper
        # levels (down_layers[2:], agg_layers[:ns-3]) do not affect the result; skipped (exact).
        skip = x
        d0, d1 = hg.down_layers[1]
        low = att_or_generic(d1, d1.conv(_canon(bconv(d0, skip))), hg.feature_atts[1], fl[2], fr[2])
        i = ns - 3
        seq = hg.agg_layers[i]
        w = seq[0].conv.weight
        ca = low.shape[1]
        ya = interp3(F.conv3d(_canon(low), w[:, :ca]), skip.shape)
        low = None
        y = ya.add_(F.conv3d(_canon(skip), w[:, ca:]))
        skip = ya = x = None
        y = norm_act3(seq[0], y)
        y = bconv(seq[1], y)
        x = att_or_generic(seq[2], seq[2].conv(_canon(y)), hg.feature_atts_up[i], fl[ns - 2 - i], fr[ns - 2 - i])
        y = None
    yb = F.conv3d(_canon(x), P["w0b"])                      # low-res branch of the final 1x1 conv
    del x
    yb = yb.permute(0, 2, 3, 4, 1).contiguous()[0]          # [Di][Hi][Wi][8]

    # ---- full-resolution tail
    at = hg.final_feature_atts_up
    la = torch.sigmoid(at.feat_att_left(fl[0]))[0].permute(1, 2, 0).contiguous().half()     # [H][W][8]
    ra = torch.sigmoid(at.feat_att_right(fr[0]))[0].permute(2, 1, 0).contiguous().half()    # [D][H][8]
    assert la.shape == (H, W, 8) and ra.shape == (D, H, 8)
    st = torch.zeros((3, 16), device=dev, dtype=torch.float64)
    e0, e1, e2 = P["eps"]
    o = torch.empty((D, H, W, 2), device=dev, dtype=torch.float16)
    w0a = P["w0a"]
    if not stream:
        y0 = torch.empty((D, H, W, 8), device=dev, dtype=torch.float16)
        mvol(v, lm32, rm32, dims, 0, D, 1, w0a, yb, y0, 0, st[0])
        z1 = torch.empty_like(y0)
        conv3(y0, 0, dims, 0, D, st[0], e0, P["wp1"], z1, 0, 8, st[1])
        z2 = y0
        conv3(z1, 0, dims, 0, D, st[1], e1, P["wp2"], z2, 0, 8, st[2])
        del z1
        conv3(z2, 0, dims, 0, D, st[2], e2, P["wpc"], o, 0, 2, None, la, ra)
        del z2, y0
    else:
        # pass 1: y0 statistics
        mvol(v, lm32, rm32, dims, 0, D, 1, w0a, yb, None, 0, st[0])

        def y0buf(a, b):
            a, b = max(0, a), min(D, b)
            t = torch.empty((b - a, H, W, 8), device=dev, dtype=torch.float16)
            mvol(v, lm32, rm32, dims, a, b, 1, w0a, yb, t, a, None)
            return t, a

        def zbuf(src, wp, st_in, eps, a, b):
            a, b = max(0, a), min(D, b)
            t = torch.empty((b - a, H, W, 8), device=dev, dtype=torch.float16)
            conv3(src[0], src[1], dims, a, b, st_in, eps, wp, t, a, 8, None)
            return t, a
        for a in range(0, D, chunk):                        # pass 2: z1 statistics
            b = min(D, a + chunk)
            y = y0buf(a - 1, b + 1)
            conv3(y[0], y[1], dims, a, b, st[0], e0, P["wp1"], None, 0, 8, st[1])
        for a in range(0, D, chunk):                        # pass 3: z2 statistics
            b = min(D, a + chunk)
            y = y0buf(a - 2, b + 2)
            z = zbuf(y, P["wp1"], st[0], e0, a - 1, b + 1)
            del y
            conv3(z[0], z[1], dims, a, b, st[1], e1, P["wp2"], None, 0, 8, st[2])
        for a in range(0, D, chunk):                        # pass 4: classifier output
            b = min(D, a + chunk)
            y = y0buf(a - 3, b + 3)
            z = zbuf(y, P["wp1"], st[0], e0, a - 2, b + 2)
            del y
            z = zbuf(z, P["wp2"], st[1], e1, a - 1, b + 1)
            conv3(z[0], z[1], dims, a, b, st[2], e2, P["wpc"], o, 0, 2, None, la, ra)
    oo = o.unsqueeze(0).permute(0, 4, 1, 2, 3).permute(0, 1, 3, 4, 2)   # (1, 2, H, W2, W3) = upstream layout
    disp = oo[:, :1].float().contiguous()   # contiguous fp32 disparity volume: corr pyramid rows
    conf = oo[:, 1:]
    # left / right disparity + confidence maps in two kernels (upstream: softmax / expectation / entropy over fp32 volumes),
    # attached to the returned volumes; fast.py's estimate_* wrappers return them
    dl = torch.empty((1, 1, H, W), device=dev, dtype=torch.float32)
    cl = torch.empty_like(dl)
    dr = torch.empty((1, 1, H, D), device=dev, dtype=torch.float32)
    cr = torch.empty_like(dr)
    k = _k()
    k.launch("est_left", (((H * W + 255) // 256, 1, 1), (256, 1, 1)), o, D, H, W, dl, cl)
    k.launch("est_right", (((D * H * 32 + 255) // 256, 1, 1), (256, 1, 1)), o, D, H, W, dr, cr)
    disp._sa_est = (dl, dr)
    conf._sa_est = (cl, cr)
    return disp, conf
