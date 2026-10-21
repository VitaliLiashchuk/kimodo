"""LoRA fine-tuning of a Kimodo checkpoint on (motion, caption) pairs.

Recipe after "Controllable Exaggeration for Generative Motion Models" (Zamani et al. 2026): x0-prediction with a
masked Huber loss on Kimodo's own motion features, LoRA on the feed-forward, motion in/out and text projections,
AdamW 1e-4, classifier-free-guidance caption dropout, EMA. Captions come pre-encoded (cache .pt from the
kimodo_tools encoder daemon) so no LLM is loaded here.

    python -m kimodo.scripts.train_lora --manifest pairs/manifest.json --emb text_emb.pt --out runs/r1 --steps 2000

Manifest: {"rows": [{"npz": path, "caption": str, ...}, ...]}; npz = Kimodo motion npz (77 or 30 joints).
"""
import argparse
import sys
import json
import re
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, inject_adapter_in_model

from kimodo import load_model
from kimodo.sanitize import sanitize_text

LORA_TARGETS = ["input_linear", "output_linear", "embed_text", "linear1", "linear2"]   # paper: FF + motion in/out + text proj


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--emb", required=True, help="caption embedding cache (.pt dict caption -> [1, 4096])")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Kimodo-SOMA-RP-v1.2")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--cfg_dropout", type=float, default=0.1)
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--overfit", type=int, default=0, help="use only the first N pairs (2N rows); smoke test")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--resume", default="")
    p.add_argument("--style_hidden", type=int, default=0, help="dedicated style input (kimodo_tools/style_cond): scalar 'level' per row added to the time token; 0 = off")
    p.add_argument("--style_lr", type=float, default=1e-3)
    p.add_argument("--learn_tag", type=float, default=0.0, help="LR for a trainable tag vector (textual inversion): rows whose caption contains the manifest tag get caption_embedding(plain) + delta, delta learned")
    p.add_argument("--prior", type=float, default=0.0, help="prior preservation weight on style-0 rows (see --prior_mode)")
    p.add_argument("--prior_mode", default="replace", choices=["replace", "add", "data"], help="data: style-0 rows = real clips, ground-truth target, weight=prior (LoRA-MDM); replace: style-0 rows target the frozen base prediction (weight=prior); add: all rows target ground truth, plus prior*huber(pred, base_pred) on style-0 rows")
    p.add_argument("--paired", action="store_true", help="batches of twin pairs (same key, style 0 and 1) sharing noise and timestep")
    p.add_argument("--prior_rows", type=int, default=0, help="with --paired: this many batch slots are random UNPAIRED natural rows (free prior), the rest twin pairs")
    p.add_argument("--targets", default=",".join(LORA_TARGETS), help="comma list of LoRA target module names (add self_attn for attention)")
    p.add_argument("--loss", default="huber", choices=["huber", "official"], help="official = tech-report per-block smooth-L1 weights (+ --fk_weight)")
    p.add_argument("--fk_weight", type=float, default=5.0, help="gamma_7: FK(pred rotations) vs target joint positions (only with --loss official)")
    p.add_argument("--balanced", action="store_true", help="each batch = half style rows, half prior (style 0) rows")
    p.add_argument("--dim_weights", default="", help=".pt with 'weights' [D]: per-channel loss weights (focus on channels the style changes)")
    p.add_argument("--constraints", default="", choices=["", "feet"], help="feet: train with the eval-time constraint mask (feet every 2 frames, root path + heading, full body on first/last frame), observed values from the natural twin")
    p.add_argument("--constraint_dropout", type=float, default=0.1, help="rows trained without constraints (keeps the unconditional CFG branch sane)")
    return p.parse_args()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PairDataset:
    """Clips as UNNORMALIZED Kimodo features (computed once with the model's motion_rep) + caption embeddings."""

    def __init__(self, manifest, emb_path, motion_rep, skeleton, overfit=0):
        man = json.load(open(manifest)); rows = man["rows"]; self.tag = man.get("tag", "")
        if overfit:
            rows = rows[: 2 * overfit]
        cache = torch.load(emb_path)
        self.items = []
        t0 = time.time()
        for r in rows:
            d = np.load(r["npz"])
            local = torch.tensor(d["local_rot_mats"], dtype=torch.float)
            if local.shape[1] != skeleton.nbjoints:
                local = skeleton.from_SOMASkeleton77(local)
            root = torch.tensor(d["root_positions"], dtype=torch.float)
            with torch.no_grad():
                feat = motion_rep(local[None], root[None], to_normalize=False)[0]      # [T, D], raw units
            cap = sanitize_text(r["caption"])
            if cap not in cache:
                raise KeyError(f"caption not in embedding cache: {cap!r}")
            tagged = bool(self.tag) and self.tag in r["caption"]
            plain = sanitize_text(re.sub(r"\s+\.", ".", re.sub(r"\s*,?\s*" + re.escape(self.tag), "", r["caption"]))) if tagged else cap
            self.items.append({"feat": feat, "text": cache[cap].reshape(1, -1).float(), "text_plain": cache[plain].reshape(1, -1).float() if plain in cache else None,
                               "tagged": tagged, "caption": cap, "key": r.get("key", ""), "style": int(r.get("style", 0)), "level": float(r.get("level", r.get("style", 0)))})
        groups = {}
        for i, it in enumerate(self.items):
            groups.setdefault(it["key"], {0: None, 1: []})
            if it["level"] == 0: groups[it["key"]][0] = i
            else: groups[it["key"]][1].append(i)
        self.pairs = [v for v in groups.values() if v[0] is not None and v[1]]
        self.prior_rows = 0; self.n_paired = 0
        paired_ids = {i for v in self.pairs for i in [v[0]] + list(v[1])}
        self.extra = [it for j, it in enumerate(self.items) if j not in paired_ids and it["style"] == 0]   # naturals without a styled twin
        self.dim = self.items[0]["feat"].shape[-1]
        print(f"dataset: {len(self.items)} rows, {sum(i['feat'].shape[0] for i in self.items)} frames, "
              f"{len(set(i['caption'] for i in self.items))} captions, features in {time.time()-t0:.0f}s", flush=True)

    def sample(self, batch, max_frames, rng, paired=False, balanced=False):
        """Returns feats, texts, styles. Paired: consecutive items are the two twins of one clip, same window."""
        if paired:
            picks, wins = [], []
            n_extra = min(self.prior_rows, len(self.extra)) if self.extra else 0
            self.n_paired = ((batch - n_extra) // 2) * 2
            for _ in range(self.n_paired // 2):
                pr = self.pairs[rng.randrange(len(self.pairs))]
                a, b = self.items[pr[0]], self.items[pr[1][rng.randrange(len(pr[1]))]]
                s = rng.randrange(0, max(a["feat"].shape[0] - max_frames, 0) + 1)
                picks += [a, b]; wins += [s, s]
            for _ in range(batch - self.n_paired):                               # unpaired prior rows: own window, own noise
                it = self.extra[rng.randrange(len(self.extra))]
                picks.append(it); wins.append(rng.randrange(0, max(it["feat"].shape[0] - max_frames, 0) + 1))
        elif balanced:                                                   # half style rows, half prior (style 0) rows
            st = [i for i in self.items if i["style"] == 1]; nt = [i for i in self.items if i["style"] == 0]
            picks = [st[rng.randrange(len(st))] for _ in range(batch // 2)] + [nt[rng.randrange(len(nt))] for _ in range(batch - batch // 2)]
            wins = [rng.randrange(0, max(it["feat"].shape[0] - max_frames, 0) + 1) for it in picks]
        else:
            picks = [self.items[rng.randrange(len(self.items))] for _ in range(batch)]
            wins = [rng.randrange(0, max(it["feat"].shape[0] - max_frames, 0) + 1) for it in picks]
        feats, texts, styles = [], [], []
        self.last_tagged, self.last_plain, self.last_levels = [], [], []
        for it, s in zip(picks, wins):
            feats.append(it["feat"][s : s + max_frames]); texts.append(it["text"]); styles.append(it["style"]); self.last_levels.append(it["level"])
            self.last_tagged.append(it["tagged"]); self.last_plain.append(it["text_plain"] if it["text_plain"] is not None else it["text"])
        return feats, texts, styles


def collate(feats, texts, motion_rep, device, cfg_dropout, rng, paired=False):
    """Random-heading canonicalization per clip (training augmentation), normalize, pad, caption dropout."""
    B = len(feats); T = max(f.shape[0] for f in feats); D = feats[0].shape[-1]
    x0 = torch.zeros(B, T, D); pad = torch.zeros(B, T, dtype=torch.bool); heading = torch.zeros(B)
    for i, f in enumerate(feats):
        if i < paired and i % 2 == 1:
            torch.manual_seed(seed_i)                                        # twin gets the same random heading
        seed_i = rng.randrange(1 << 30)
        if i < paired and i % 2 == 0:
            torch.manual_seed(seed_i)
        canon = motion_rep.canonicalize_random_heading(f[None])              # frame 0 at origin, random heading
        heading[i] = motion_rep.get_first_heading_angle(canon, normalized=False)[0]
        x0[i, : f.shape[0]] = motion_rep.normalize(canon)[0]
        pad[i, : f.shape[0]] = True
    text = torch.cat(texts, 0)[:, None, :]                                   # [B, 1, 4096]
    tmask = torch.ones(B, 1, dtype=torch.bool)
    for i in range(B):
        if rng.random() < cfg_dropout:                                       # unconditional branch = zero text, mask off
            text[i] = 0; tmask[i] = False
    return x0.to(device), pad.to(device), text.to(device), tmask.to(device), heading.to(device)


def constraint_columns(motion_rep, skeleton, npz):
    """Feature columns each make_feet_keys constraint type writes (fullbody / foot / root2d), read off Kimodo's own
    encoder on a 4-frame stub: frame 0 = all three, frame 1 = root2d only, frame 2 = root2d + feet."""
    from scipy.spatial.transform import Rotation as Rot
    from kimodo.constraints import load_constraints_lst
    d = np.load(npz); J = d["local_rot_mats"].shape[1]
    def block(kind, idx):
        aa = Rot.from_matrix(d["local_rot_mats"][idx].reshape(-1, 3, 3)).as_rotvec().reshape(len(idx), J, 3)
        return {"type": kind, "frame_indices": idx, "local_joints_rot": aa.tolist(), "root_positions": d["root_positions"][idx].tolist(),
                "smooth_root_2d": d["smooth_root_pos"][idx][:, [0, 2]].tolist()}
    sets = [block("fullbody", [0]), block("left-foot", [2]), block("right-foot", [2]),
            {"type": "root2d", "frame_indices": [0, 1, 2, 3], "smooth_root_2d": d["smooth_root_pos"][:4][:, [0, 2]].tolist(),
             "global_root_heading": d["global_root_heading"][:4].tolist()}]
    with torch.no_grad():
        _, m = motion_rep.create_conditions_from_constraints(load_constraints_lst(sets, skeleton), 4, True, "cpu")
    root = m[1] > 0; foot = (m[2] > 0) & ~root; fb = m[0] > 0
    return {k: v.nonzero().squeeze(1) for k, v in (("root", root), ("foot", foot), ("fb", fb))}


def constraint_mask(cols, pad, D, n_paired, rng, p_drop):
    """[B, T, D] eval-style mask per row (root all frames, feet every 2nd frame, full body first + last); twins share it."""
    B, T = pad.shape
    m = torch.zeros(B, T, D)
    for i in range(B):
        if i < n_paired and i % 2 == 1:
            m[i] = m[i - 1]; continue
        if rng.random() < p_drop:
            continue
        L = int(pad[i].sum())
        m[i, :L, cols["root"]] = 1
        m[i, torch.arange(0, L, 2)[:, None], cols["foot"][None, :]] = 1
        m[i, torch.tensor([0, L - 1])[:, None], cols["fb"][None, :]] = 1
    return m


def lora_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}


BLOCK_W = {"smooth_root_pos": 10.0, "global_root_heading": 2.0, "local_joints_positions": 10.0,
           "global_rot_data": 10.0, "velocities": 3.0, "foot_contacts": 4.0}        # Kimodo tech report, gamma_1..6


def official_loss(pred, target, pad, row_w, motion_rep, skeleton, fk_w):
    """Kimodo tech-report loss: per-block smooth-L1 with the paper's weights (+ FK(rot) vs positions, gamma_7).
    pred/target are NORMALISED features [B, T, D]; pad [B, T]; row_w [B, 1]."""
    from kimodo.geometry import cont6d_to_matrix
    from kimodo.skeleton.kinematics import joints_from_global_rots
    rw = row_w[:, 0][:, None] * pad; den = pad.sum()
    tot, wsum = 0.0, 0.0
    for k, w in BLOCK_W.items():
        sl = motion_rep.slice_dict[k]
        per = F.smooth_l1_loss(pred[..., sl], target[..., sl], reduction="none").mean(-1)
        tot = tot + w * (per * rw).sum() / den; wsum += w
    if fk_w > 0:
        B, T, _ = pred.shape; J = skeleton.nbjoints
        un_p = motion_rep.unnormalize(pred); un_t = motion_rep.unnormalize(target)
        R = cont6d_to_matrix(un_p[..., motion_rep.slice_dict["global_rot_data"]].reshape(B, T, J, 6))
        _, rel_p = joints_from_global_rots(R, torch.zeros(B, T, 3, device=pred.device, dtype=pred.dtype), skeleton, return_norootpos=True)
        gt = un_t[..., motion_rep.slice_dict["local_joints_positions"]].reshape(B, T, J, 3)
        rel_t = gt - gt[:, :, skeleton.root_idx : skeleton.root_idx + 1]
        pos_std = motion_rep.stats.std[motion_rep.slice_dict["local_joints_positions"]].to(pred.device, pred.dtype).mean()
        per = F.smooth_l1_loss(rel_p / pos_std, rel_t / pos_std, reduction="none").mean((-1, -2))
        tot = tot + fk_w * (per * rw).sum() / den; wsum += fk_w
    return tot / wsum


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed); torch.manual_seed(args.seed)
    device = pick_device(args.device)
    model = load_model(args.model, device="cpu", text_encoder=object())
    backbone = model.denoiser.model                                          # TwostageDenoiser (no CFG wrapper)
    motion_rep, skeleton, diffusion = model.motion_rep, model.skeleton, model.diffusion
    use_t, _ = diffusion.space_timesteps(diffusion.num_base_steps)
    diffusion.calc_diffusion_vars(use_t)                                     # full 1000-step schedule for training

    data = PairDataset(args.manifest, args.emb, motion_rep, skeleton, overfit=args.overfit)

    data.prior_rows = args.prior_rows
    base = None
    if args.prior > 0 and args.prior_mode != "data":
        import copy
        base = copy.deepcopy(backbone).eval()                                # frozen base model = prior
        for p_ in base.parameters():
            p_.requires_grad_(False)

    targets = args.targets.split(",")
    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, target_modules=targets, bias="none")
    backbone = inject_adapter_in_model(cfg, backbone)
    for n, p in backbone.named_parameters():
        p.requires_grad_("lora_" in n)
    trainable = [p for p in backbone.parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in trainable); n_all = sum(p.numel() for p in backbone.parameters())
    print(f"device {device} | LoRA params {n_tr/1e6:.2f}M of {n_all/1e6:.1f}M | targets {targets}", flush=True)
    if args.style_hidden > 0:                                                # install before the device move
        sys.path.insert(0, "/Users/Shared/coding/animation_model/kimodo_tools"); import style_cond
        _sp = style_cond.install(backbone, args.style_hidden)
    for mod in ((backbone, diffusion, base) if base is not None else (backbone, diffusion)):   # MPS has no float64: cast schedule buffers
        for name, buf in list(mod.named_buffers()):
            if buf.dtype == torch.float64:
                obj, attr = mod, name.split(".")
                for a in attr[:-1]:
                    obj = getattr(obj, a)
                setattr(obj, attr[-1], buf.float())
    backbone.to(device); diffusion.to(device); motion_rep.to("cpu")
    if base is not None:
        base.to(device)
    backbone.train()

    style_params = []
    if args.style_hidden > 0:
        sys.path.insert(0, "/Users/Shared/coding/animation_model/kimodo_tools"); import style_cond
        style_params = [p_ for bb_ in (backbone.root_model, backbone.body_model) for p_ in bb_.embed_timestep.proj.parameters()]
        for p_ in style_params: p_.requires_grad_(True)
        print(f"style input installed: {sum(p_.numel() for p_ in style_params)/1e3:.0f}k params, lr {args.style_lr}", flush=True)
    tag_delta = None
    if args.learn_tag > 0:                                                   # textual inversion: tagged caption = plain + delta
        ex = next(i for i in data.items if i["tagged"])
        tag_delta = torch.nn.Parameter((ex["text"] - ex["text_plain"]).reshape(-1).to(device))
        print(f"learnable tag vector: init |delta| {tag_delta.norm():.1f} (|caption| ~{ex['text'].norm():.0f}), lr {args.learn_tag}", flush=True)
    opt = torch.optim.AdamW([{"params": trainable, "lr": args.lr}] + ([{"params": [tag_delta], "lr": args.learn_tag, "weight_decay": 0.0}] if tag_delta is not None else [])
                            + ([{"params": style_params, "lr": args.style_lr, "weight_decay": 0.0}] if style_params else []), lr=args.lr, weight_decay=0.0)
    ema = {k: v.clone() for k, v in lora_state(backbone).items()}
    step0 = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu")
        backbone.load_state_dict(ck["lora"], strict=False); ema = ck["ema"]; opt.load_state_dict(ck["opt"]); step0 = ck["step"]
        print(f"resumed from {args.resume} at step {step0}", flush=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    log = open(os.path.join(args.out, "train.log"), "a")

    ccols = None
    if args.constraints == "feet":
        ccols = constraint_columns(motion_rep, skeleton, json.load(open(args.manifest))["rows"][0]["npz"])
        print(f"constraint-matched training: root {len(ccols['root'])} cols all frames, feet {len(ccols['foot'])} cols every 2nd frame, "
              f"full body {len(ccols['fb'])} cols first/last frame, dropout {args.constraint_dropout}", flush=True)
    dim_w = torch.load(args.dim_weights)["weights"].float().to(device) if args.dim_weights else None
    if dim_w is not None:
        dim_w = dim_w / dim_w.mean()                                         # keep the loss scale comparable
        print(f"per-channel loss weights: max {dim_w.max():.2f}, {(dim_w > 2).sum().item()} channels > 2x", flush=True)
    t_start = time.time(); run_loss = 0.0; n_loss = 0
    for step in range(step0 + 1, args.steps + 1):
        feats, texts, styles = data.sample(args.batch, args.max_frames, rng, paired=args.paired, balanced=args.balanced)
        n_paired = data.n_paired if args.paired else 0
        x0, pad, text, tmask, heading = collate(feats, texts, motion_rep, device, args.cfg_dropout, rng, paired=n_paired)
        B, T, D = x0.shape
        if style_params:
            style_cond.set_level(backbone, data.last_levels)
        if tag_delta is not None:                                            # rebuild tagged captions from plain + learnable delta
            for i in range(B):
                if data.last_tagged[i] and tmask[i, 0]:
                    text[i, 0] = data.last_plain[i].reshape(-1).to(device) + tag_delta
        t = torch.randint(0, diffusion.num_base_steps, (B,), device=device)
        noise = torch.randn_like(x0)
        if args.paired:                                                      # twins share timestep and noise (paired prefix only)
            t[1:n_paired:2] = t[0:n_paired:2]; noise[1:n_paired:2] = noise[0:n_paired:2]
        xt = diffusion.q_sample(x0, t, noise)
        cmask = torch.zeros_like(x0)                                         # kinematic constraints: none, or the eval mask
        if ccols is not None:
            cmask = constraint_mask(ccols, pad.cpu(), D, n_paired, rng, args.constraint_dropout).to(device)
        obs = x0.clone()
        if n_paired:
            obs[1:n_paired:2] = x0[0:n_paired:2]                             # style twin observes the NATURAL clip (as at eval)
        obs = obs * cmask
        pred = backbone(xt, pad, text, tmask, t, first_heading_angle=heading, motion_mask=cmask, observed_motion=obs)
        target = x0; row_w = torch.ones(B, 1, device=device)
        if base is not None:                                                 # prior preservation: natural rows -> base prediction
            st = torch.tensor(styles, device=device)
            with torch.no_grad():
                base_pred = base(xt, pad, text, tmask, t, first_heading_angle=heading, motion_mask=cmask, observed_motion=obs)
            nat = (st == 0)[:, None, None]
            if args.prior_mode == "replace":
                target = torch.where(nat, base_pred, x0); row_w = torch.where(nat[:, :, 0], torch.full_like(row_w, args.prior), row_w)
        if args.prior > 0 and args.prior_mode == "data":                     # real natural rows, ground truth, weight lambda
            st = torch.tensor(styles, device=device); row_w = torch.where((st == 0)[:, None], torch.full_like(row_w, args.prior), row_w)
        if args.loss == "official":
            loss = official_loss(pred, target, pad, row_w, motion_rep, skeleton, args.fk_weight)
        else:
            per = F.huber_loss(pred, target, reduction="none", delta=1.0)        # [B, T, D]
            per = (per * dim_w).mean(-1) if dim_w is not None else per.mean(-1)   # [B, T]
            loss = (per * pad * row_w).sum() / pad.sum()
        if base is not None and args.prior_mode == "add":                    # extra pull toward the base on natural rows
            pp = F.huber_loss(pred, base_pred, reduction="none", delta=1.0).mean(-1) * nat[:, :, 0]
            loss = loss + args.prior * (pp * pad).sum() / pad.sum()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable + ([tag_delta] if tag_delta is not None else []) + style_params, 1.0)
        opt.step()
        with torch.no_grad():
            for k, v in lora_state(backbone).items():
                ema[k].mul_(args.ema).add_(v, alpha=1 - args.ema)
        run_loss += loss.item(); n_loss += 1
        if step % args.log_every == 0 or step == 1:
            el = time.time() - t_start
            msg = f"step {step}/{args.steps}  loss {run_loss/n_loss:.4f}  {el/(step-step0):.2f}s/step  T={T} B={B}"
            print(msg, flush=True); log.write(msg + "\n"); log.flush(); run_loss = 0.0; n_loss = 0
        if step % args.save_every == 0 or step == args.steps:
            extra = {"tag_delta": tag_delta.detach().cpu().clone(), "tag": data.tag} if tag_delta is not None else {}
            if style_params: extra.update({"style": style_cond.state(backbone), "style_hidden": args.style_hidden})
            torch.save({"step": step, "lora": lora_state(backbone), "ema": ema, "opt": opt.state_dict(), **extra,
                        "config": {"r": args.rank, "alpha": args.alpha, "targets": targets, "model": args.model}},
                       os.path.join(args.out, f"lora_step{step:06d}.pt"))
            torch.save({"step": step, "lora": ema, **extra, "config": {"r": args.rank, "alpha": args.alpha, "targets": targets, "model": args.model}},
                       os.path.join(args.out, "lora_ema_latest.pt"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
