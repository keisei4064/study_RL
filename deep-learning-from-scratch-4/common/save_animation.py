from matplotlib import animation
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional


# アニメーション
def save_animation_with_steps(
    plot_name: str,
    fig_history: list[Figure],
    filepath: str = "anim.gif",
    interval: int = 500,
    show_anim: bool = False,
    frame_names: Optional[list[str]] = None,
):
    """fig_historyからタイトル付きアニメーションを生成（軸非表示）"""

    if frame_names is None:
        frame_names = [f"Step: {i + 1}" for i in range(len(fig_history))]
    else:
        assert len(fig_history) == len(frame_names), (
            "fig_historyとframe_namesの要素数が違います"
        )

    fig = plt.figure(figsize=fig_history[0].get_size_inches(), layout="constrained")
    ax = fig.add_subplot(111)
    ax.axis("off")  # 軸を非表示
    ims = []

    for i, f in enumerate(fig_history, start=0):
        # Figure内容を画像化
        buf = io.BytesIO()
        f.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = plt.imread(buf)

        # フレーム画像
        im = ax.imshow(img, animated=True)

        # タイトルをAxes上に描画（ここが確実に残る）
        title_text = f"{plot_name} - {frame_names[i]}"
        title = ax.text(
            0.5,
            1.02,
            title_text,
            fontsize=16,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

        ims.append([im, title])
        plt.close(f)

    ani = animation.ArtistAnimation(
        fig, ims, interval=interval, repeat_delay=1000, blit=False
    )

    if filepath.endswith(".gif"):
        ani.save(filepath, writer="pillow")
    elif filepath.endswith(".mp4"):
        ani.save(filepath, writer="ffmpeg")
    else:
        raise ValueError("filename must end with .gif or .mp4")

    print(f"✅ アニメーションを保存しました: {filepath}")

    if show_anim:
        plt.show()

    plt.close(fig)
