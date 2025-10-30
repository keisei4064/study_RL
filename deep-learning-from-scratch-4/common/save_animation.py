from matplotlib import animation
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


# アニメーション
def save_animation_with_steps(
    plot_name: str,
    fig_history: list[Figure],
    filepath: str = "anim.gif",
    interval: int = 500,
    show_anim: bool = False,
):
    """fig_historyからタイトル付きアニメーションを生成（軸非表示）"""

    fig = plt.figure(figsize=fig_history[0].get_size_inches(), layout="constrained")
    ax = fig.add_subplot(111)
    ax.axis("off")  # 軸を非表示
    ims = []

    for i, f in enumerate(fig_history, start=1):
        # Figure内容を画像化
        buf = io.BytesIO()
        f.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = plt.imread(buf)

        # フレーム画像
        im = ax.imshow(img, animated=True)

        # タイトルをAxes上に描画（ここが確実に残る）
        title_text = f"{plot_name} - Step: {i}"
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
