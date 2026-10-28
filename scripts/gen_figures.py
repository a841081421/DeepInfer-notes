# -*- coding: utf-8 -*-
"""Generate all figures for the DeepInfer study notes.

Data in `dspark` figure comes from DeepInfer's doc/inference_deployment_qa.md
benchmark table (EP64/DP64). Other figures are schematic redraws based on the
same document. Run:  python scripts/gen_figures.py
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#4D6BFE"      # deepseek blue
DBLUE = "#1E2A5A"
LBLUE = "#EEF1FF"
GREEN = "#2FBF71"
ORANGE = "#FF6A3D"
RED = "#E5484D"
GRAY = "#6B7280"
LGRAY = "#F3F4F6"
YELLOW = "#FFC53D"


def box(ax, x, y, w, h, text, fc=LBLUE, ec=BLUE, tc=DBLUE, fs=10, lw=1.4, bold=False, rounded=True):
    style = "round,pad=0.02,rounding_size=0.06" if rounded else "square,pad=0.02"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, xy1, xy2, color=GRAY, lw=1.6, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}"))


# ---------------------------------------------------------------- figure 1
def fig_architecture(path):
    fig, ax = plt.subplots(figsize=(13.5, 9))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 97, "DeepInfer / V4.1-Flash 推理系统:E/P/D 分离架构", ha="center",
            fontsize=16, fontweight="bold", color=DBLUE)
    ax.text(50, 93.6, "重绘自官方部署 Q&A(doc/inference_deployment_qa.md),示意", ha="center",
            fontsize=9.5, color=GRAY)

    # client + entrypoint
    box(ax, 30, 84, 40, 6.5, "客户端请求(Chat Completions / Responses / Messages API)",
        fc="white", ec=GRAY, tc=DBLUE, fs=10.5, bold=True)
    box(ax, 30, 74, 40, 7, "Entrypoint + DeepSeek Recipe\n(Rust 编解码·前缀匹配·防特殊 token 注入)",
        fs=9.5)
    arrow(ax, (50, 84), (50, 81.2))

    # load balancers row
    box(ax, 6, 63, 24, 7, "Vision Encoder LB\n(按机器·按 patch 数)", fs=9)
    box(ax, 38, 63, 24, 7, "Prefill LB\n(按上下文长度 4 档分桶·算力配额)", fs=9)
    box(ax, 70, 63, 24, 7, "Decode LB\n(请求数 × 总 KV 长度双维均衡)", fs=9)
    arrow(ax, (44, 74), (18, 70.2), rad=-0.15)
    arrow(ax, (50, 74), (50, 70.2))
    arrow(ax, (56, 74), (82, 70.2), rad=0.15)

    # three pools
    box(ax, 3, 40, 30, 19,
        "Vision Encoder 池\n\n单卡即可·廉价 GPU\n多模态输入 → embedding",
        fc="white", fs=9.5)
    box(ax, 35, 40, 30, 19,
        "Prefill 池\n\n普通 8 卡机\nEP2–8 / DP2–8(小 EP)\n预填 KV → 写入存储",
        fc=LBLUE, fs=9.5, bold=False)
    box(ax, 67, 40, 30, 19,
        "Decode 池\n\n64–128 卡超节点\nEP64–128 Wide-EP + EPLB\nDSpark 变长投机解码",
        fc=LBLUE, fs=9.5)
    arrow(ax, (18, 63), (18, 59.4))
    arrow(ax, (50, 63), (50, 59.4))
    arrow(ax, (82, 63), (82, 59.4))

    # storage
    box(ax, 35, 22, 30, 12,
        "3FS KV Cache 存储集群\n10+2 纠删码·~100 PiB SSD\nTTL:agent 3 天 / chatbot 7 天",
        fc="#FFF7ED", ec=ORANGE, tc="#7C2D12", fs=9.5)
    box(ax, 71, 22, 26, 12,
        "SWA KV:本机内存\n20 层×134 tok×528 B ≈ 1.4 MB/seq\n与 block KV 共享 P→D 通道\n跨机命中走 400G VLAN",
        fc="white", ec=GREEN, tc="#065F46", fs=8.6)

    arrow(ax, (50, 40), (50, 34.4), color=ORANGE)
    ax.text(51.2, 37, "写入 / 读回(异步 fetch + GDR 上载)", fontsize=8.5, color="#7C2D12", ha="left")

    # bottom facts
    box(ax, 3, 8, 94, 9,
        "生产参数:单集群 155 万亿 token/天 · 每 GPU 时营收 $38(谷时电价)· 存储按 14.4 TB/GPU · 互联 400 Gbps/8 GPU\n"
        "最小高效部署单元:64 GPU Prefill + 64 GPU Decode(P:D ≈ 1:1)· 本次开源缺:缓存管理 / 分离部署 / 全局负载均衡",
        fc=LGRAY, ec=GRAY, tc=DBLUE, fs=9.5)

    # async scheduling note
    box(ax, 3, 22, 28, 12,
        "异步调度(零开销)\nGPU: forward(N)→sample(N)\nCPU: schedule(N+1) ∥ post(N−1)",
        fc="white", ec=BLUE, fs=8.8)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def fig_dspark(path):
    conc = [16, 32, 48, 60, 72, 96]
    tput_fixed = [4891, 8964, 12032, 13730, 14962, 16817]
    tput_dyn = [4816, 9111, 12467, 14466, 15961, 18660]
    gain = [(d / f - 1) * 100 for d, f in zip(tput_dyn, tput_fixed)]
    verified = [4.44, 3.65, 3.94, 3.58, 3.24, 3.24]
    accepted_fixed = [3.32, 3.25, 3.30, 3.30, 3.29, 3.30]
    accepted_dyn = [3.24, 3.17, 3.26, 3.21, 3.12, 3.15]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.plot(conc, tput_fixed, "o-", color=GRAY, lw=2.2, ms=6, label="固定验证 5 个 draft token")
    ax1.plot(conc, tput_dyn, "o-", color=BLUE, lw=2.4, ms=6, label="DSpark 变长验证(置信度调度)")
    for x, y, g in zip(conc, tput_dyn, gain):
        ax1.annotate(f"+{g:.1f}%" if g >= 0 else f"{g:.1f}%", (x, y),
                     textcoords="offset points", xytext=(4, 10), fontsize=9,
                     color=GREEN if g >= 0 else RED, fontweight="bold")
    ax1.set_xlabel("并发(序列数 / rank)", fontsize=11)
    ax1.set_ylabel("解码吞吐(tok/s/rank)", fontsize=11)
    ax1.set_title("吞吐:生产工作点 +5.4%,高并发 +11%", fontsize=12.5, color=DBLUE, fontweight="bold")
    ax1.legend(fontsize=10, frameon=False)
    ax1.grid(alpha=0.3)
    ax1.set_ylim(4000, 20500)

    import numpy as np
    x = np.arange(len(conc))
    w = 0.27
    ax2.bar(x - w, [5.0] * 6, w, color=LGRAY, ec=GRAY, label="固定:每步验证 token 数(恒为 5)")
    ax2.bar(x, verified, w, color=BLUE, label="变长:每步实际验证 token 数")
    ax2.bar(x + w, accepted_dyn, w, color=GREEN, alpha=0.85, label="变长:平均接受长度(几乎不变)")
    ax2.axhline(3.2, color=ORANGE, ls="--", lw=1.4)
    ax2.text(5.35, 3.28, "接受长度\n≈ 3.2", fontsize=8.5, color=ORANGE, ha="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(c) for c in conc])
    ax2.set_xlabel("并发(序列数 / rank)", fontsize=11)
    ax2.set_ylabel("token 数 / 步", fontsize=11)
    ax2.set_title("验证量 −30%(5.0 → 3.2–4.4),接受长度几乎不掉", fontsize=12.5, color=DBLUE, fontweight="bold")
    ax2.legend(fontsize=8.8, frameon=False, loc="upper center", ncol=1)
    ax2.grid(alpha=0.3, axis="y")
    ax2.set_ylim(0, 6.2)

    fig.suptitle("DSpark:Confidence Head 预测接受率,只验证边际收益最高的前缀子集(EP64/DP64 官方基准)",
                 fontsize=12, color=GRAY, y=1.02)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def fig_kv(path):
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    ax.text(50, 96, "KV Cache 三级体系:缓存命中就是钱", ha="center", fontsize=16,
            fontweight="bold", color=DBLUE)

    # tier 1
    box(ax, 4, 70, 92, 17,
        "① GPU 显存(HBM)   在途 block KV + state KV\n"
        "decode 不做 KV 卸载:瓶颈不在容量;decode 产生的 KV 仍写盘以提高命中率",
        fc=LBLUE, ec=BLUE, fs=10.5, bold=False)
    # tier 2
    box(ax, 4, 45, 92, 17,
        "② 本机内存   SWA KV(state cache 每序列一个定长槽)\n"
        "仅 20 层 × 134 token × 528 B(MXFP8)≈ 1.4 MB/seq · 单副本不进 3FS · 逻辑统一内存目录登记归属,跨机命中 400G VLAN 拉回",
        fc="white", ec=GREEN, tc="#065F46", fs=10)
    # tier 3
    box(ax, 4, 20, 92, 17,
        "③ 3FS 分布式存储集群   分页 block KV(主 KV + 索引 K)\n"
        "10+2 纠删码(11 节点并发读,任意 10 个即可解码)· ~100 PiB SSD · LRU GC · TTL:agent 3 天 / chatbot 7 天",
        fc="#FFF7ED", ec=ORANGE, tc="#7C2D12", fs=10)

    arrow(ax, (50, 70), (50, 62.4), color=GREEN, lw=2)
    ax.text(51.5, 66, "swap-in:独立进程异步 fetch → 共享内存 → GDR(RDMA)上载,与计算重叠", fontsize=9, color="#065F46")
    arrow(ax, (50, 45), (50, 37.4), color=ORANGE, lw=2)
    ax.text(51.5, 41, "命中判定:Entrypoint 对元数据做哈希链前缀匹配(仅元数据,不动 KV)", fontsize=9, color="#7C2D12")

    # right side facts
    box(ax, 4, 4, 92, 11,
        "命中率经济学:TTL 从 7 天砍到 3 天,chatbot 损失 0.5%–1% 命中率(长尾:少量缓存被读极多次)\n"
        "压缩:主 KV per-16 e4m3(NVFP4 无全局 scale)· 索引 KV per-32 ue8m0 MXFP4 · MoE 路由专家权重 MXFP4\n"
        "SWA bounded replay:缓存缺失时只重算最近 128 token,注意力窗口截断在重放段内 → 重算代价 = 常数",
        fc=LGRAY, ec=GRAY, fs=9.5)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
def fig_context(path):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.6), height_ratios=[1, 1])
    fig.suptitle("Agent 上下文布局:决定你的前缀缓存命中率(也决定 API 账单)", fontsize=15,
                 fontweight="bold", color=DBLUE, y=0.99)

    def strip(ax, x, w, label, color, tc="white", fs=9):
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 0.42, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc=color, ec="white", lw=1.2))
        ax.text(x + w / 2, 0.56, label, ha="center", va="center", fontsize=fs, color=tc)

    # bad
    ax = axes[0]
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.15); ax.axis("off")
    ax.text(0, 1.02, "× 坏布局:易变内容插在中间 → 插入点之后全部缓存失效", fontsize=12, color=RED, fontweight="bold")
    strip(ax, 1, 14, "System Prompt", GREEN)
    strip(ax, 15.5, 14, "工具定义", GREEN)
    strip(ax, 26, 10, "第 1 轮", GREEN)
    strip(ax, 36.5, 11, "时间戳", RED)
    strip(ax, 48, 10, "第 2 轮", LGRAY, tc=DBLUE)
    strip(ax, 58.5, 12, "检索结果", RED)
    strip(ax, 71, 10, "第 3 轮", LGRAY, tc=DBLUE)
    strip(ax, 81.5, 10, "运行状态", RED)
    strip(ax, 92, 7, "新输入", YELLOW, tc=DBLUE)
    ax.annotate("缓存命中", (20, 0.83), fontsize=9.5, color=GREEN, ha="center", fontweight="bold")
    ax.annotate("", xy=(35.5, 0.79), xytext=(1.5, 0.79), arrowprops=dict(arrowstyle="|-|", color=GREEN, lw=1.6))
    ax.annotate("全部重算(每一轮都重算)", (66, 0.22), fontsize=9.5, color=RED, ha="center", fontweight="bold")
    ax.annotate("", xy=(92, 0.30), xytext=(37, 0.30), arrowprops=dict(arrowstyle="|-|", color=RED, lw=1.6))

    # good
    ax = axes[1]
    ax.set_xlim(0, 100); ax.set_ylim(0, 1.15); ax.axis("off")
    ax.text(0, 1.02, "√ 好布局:稳定前缀字节级不变,易变内容只追加在尾部", fontsize=12, color=GREEN, fontweight="bold")
    strip(ax, 1, 14, "System Prompt\n(冻结)", GREEN)
    strip(ax, 15.5, 14, "工具定义\n(冻结)", GREEN)
    strip(ax, 30, 12, "第 1 轮", GREEN)
    strip(ax, 42.5, 12, "第 2 轮", GREEN)
    strip(ax, 55, 12, "第 3 轮", GREEN)
    strip(ax, 67.5, 10, "易变信息\n统一放尾部", YELLOW, tc=DBLUE, fs=8)
    strip(ax, 78, 8, "新输入", ORANGE)
    strip(ax, 86.5, 12, "(供下一轮续写)", "white", tc=GRAY, fs=8)
    ax.annotate("前缀缓存命中(按 API 缓存价计费,约为原价 1/10)", (42, 0.16), fontsize=9.5, color=GREEN, ha="center", fontweight="bold")
    ax.annotate("", xy=(65.5, 0.22), xytext=(1.5, 0.22), arrowprops=dict(arrowstyle="|-|", color=GREEN, lw=1.6))
    ax.annotate("仅此段\n按原价", (78.5, 0.80), fontsize=9, color=ORANGE, ha="center")

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    import os
    os.makedirs("images", exist_ok=True)
    fig_architecture("images/architecture.png")
    fig_dspark("images/dspark_throughput.png")
    fig_kv("images/kv_cache_tiers.png")
    fig_context("images/agent_context_design.png")
    print("done:", os.listdir("images"))
