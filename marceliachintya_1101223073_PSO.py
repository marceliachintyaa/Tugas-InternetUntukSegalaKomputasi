import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def f(x):
    return x**2


def run_pso(
    w,
    c1=1.5,
    c2=1.5,
    num_particles=20,
    max_iterations=100,
    threshold=1.0,
    target_ratio=0.90,
    seed=None,
    keep_history=False,
):
    rng = np.random.default_rng(seed)

    positions = rng.uniform(-10, 10, num_particles)
    velocities = rng.uniform(-1, 1, num_particles)

    pbest_positions = positions.copy()
    pbest_values = f(positions)

    gbest_index = np.argmin(pbest_values)
    gbest_position = pbest_positions[gbest_index]

    history = []
    if keep_history:
        history.append(
            {
                "iteration": 0,
                "positions": positions.copy(),
                "gbest_position": gbest_position,
                "ratio_near": np.mean(np.abs(positions) <= threshold),
                "mean_abs_position": np.mean(np.abs(positions)),
            }
        )

    target_iteration = None

    for iteration in range(1, max_iterations + 1):
        for i in range(num_particles):
            r1 = rng.random()
            r2 = rng.random()

            velocities[i] = (
                w * velocities[i]
                + c1 * r1 * (pbest_positions[i] - positions[i])
                + c2 * r2 * (gbest_position - positions[i])
            )

            positions[i] += velocities[i]
            fitness = f(positions[i])

            if fitness < pbest_values[i]:
                pbest_values[i] = fitness
                pbest_positions[i] = positions[i]

        gbest_index = np.argmin(pbest_values)
        gbest_position = pbest_positions[gbest_index]

        ratio_near = np.mean(np.abs(positions) <= threshold)
        mean_abs_position = np.mean(np.abs(positions))

        if keep_history:
            history.append(
                {
                    "iteration": iteration,
                    "positions": positions.copy(),
                    "gbest_position": gbest_position,
                    "ratio_near": ratio_near,
                    "mean_abs_position": mean_abs_position,
                }
            )

        if target_iteration is None and ratio_near >= target_ratio:
            target_iteration = iteration
            if not keep_history:
                break

    return target_iteration, history


def run_experiment():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    num_particles = 20
    max_iterations = 100
    c1 = 1.5
    c2 = 1.5
    threshold = 1.0
    target_ratio = 0.90
    n_trials = 30
    w_values = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]

    rows = []
    base_seed = 42
    for w in w_values:
        for trial in range(1, n_trials + 1):
            target_iteration, _ = run_pso(
                w=w,
                c1=c1,
                c2=c2,
                num_particles=num_particles,
                max_iterations=max_iterations,
                threshold=threshold,
                target_ratio=target_ratio,
                seed=base_seed + int(w * 1000) + trial,
                keep_history=False,
            )
            rows.append(
                {
                    "w": w,
                    "c1": c1,
                    "c2": c2,
                    "num_particles": num_particles,
                    "max_iterations": max_iterations,
                    "trial": trial,
                    "target_iteration": target_iteration,
                    "status": "Reached" if target_iteration is not None else "Not reached",
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv("results/pso_w_results.csv", index=False)

    summary = (
        df.assign(target_iteration_numeric=df["target_iteration"].astype(float))
        .groupby("w")
        .agg(
            successful_trials=("status", lambda s: (s == "Reached").sum()),
            total_trials=("status", "count"),
            mean_iteration=("target_iteration_numeric", "mean"),
            median_iteration=("target_iteration_numeric", "median"),
            min_iteration=("target_iteration_numeric", "min"),
            max_iteration=("target_iteration_numeric", "max"),
            std_iteration=("target_iteration_numeric", "std"),
        )
        .reset_index()
    )
    summary["success_rate"] = summary["successful_trials"] / summary["total_trials"] * 100
    summary.to_csv("results/pso_w_summary.csv", index=False)

    print("\nDetailed results saved to results/pso_w_results.csv")
    print("Summary saved to results/pso_w_summary.csv\n")
    print(summary)

    plot_summary = summary.dropna(subset=["mean_iteration"]).copy()
    plt.figure(figsize=(7, 4.5))
    plt.plot(plot_summary["w"], plot_summary["mean_iteration"], marker="o")
    plt.xlabel("Inertia constant, w")
    plt.ylabel("Mean target iteration")
    plt.title("Effect of Inertia w on PSO Convergence")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/pso_w_mean_iterations.png", dpi=300)
    plt.close()

    # Plot 2: success rate
    plt.figure(figsize=(7, 4.5))
    plt.plot(summary["w"], summary["success_rate"], marker="o")
    plt.xlabel("Inertia constant, w")
    plt.ylabel("Success rate within 100 iterations (%)")
    plt.title("Success Rate of Reaching the 90% Particle Target")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig("figures/pso_w_success_rate.png", dpi=300)
    plt.close()

    # History and animation for baseline w=0.7
    target_iteration, history = run_pso(
        w=0.7,
        c1=c1,
        c2=c2,
        num_particles=num_particles,
        max_iterations=max_iterations,
        threshold=threshold,
        target_ratio=target_ratio,
        seed=123,
        keep_history=True,
    )

    hist_df = pd.DataFrame(
        [
            {
                "iteration": h["iteration"],
                "ratio_near": h["ratio_near"],
                "mean_abs_position": h["mean_abs_position"],
                "gbest_position": h["gbest_position"],
                "gbest_value": f(h["gbest_position"]),
            }
            for h in history
        ]
    )
    hist_df.to_csv("results/pso_history_w07.csv", index=False)

    plt.figure(figsize=(7, 4.5))
    plt.plot(hist_df["iteration"], hist_df["ratio_near"] * 100, label="Particles near optimum")
    plt.axhline(90, linestyle="--", label="Target = 90%")
    if target_iteration is not None:
        plt.axvline(target_iteration, linestyle=":", label=f"Reached at iter. {target_iteration}")
    plt.xlabel("Iteration")
    plt.ylabel("Particles within |x| <= 1 (%)")
    plt.title("Convergence Progress for w=0.7")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/pso_convergence_w07.png", dpi=300)
    plt.close()

    x_curve = np.linspace(-10, 10, 500)
    frame_numbers = [0, 5, 10, 20, 50, 100]
    for frame in frame_numbers:
        h = history[min(frame, len(history) - 1)]
        plt.figure(figsize=(7, 4.5))
        plt.plot(x_curve, f(x_curve), linewidth=2, label="f(x)=x^2")
        plt.scatter(h["positions"], f(h["positions"]), s=50, label="Particles")
        plt.scatter([h["gbest_position"]], [f(h["gbest_position"])], marker="x", s=120, label="Global best")
        plt.axvspan(-threshold, threshold, alpha=0.15, label="Target zone |x| <= 1")
        plt.xlim(-10, 10)
        plt.ylim(0, 100)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"PSO Particle Movement - Iteration {h['iteration']}")
        plt.legend(loc="upper center", ncol=2, fontsize=8)
        plt.tight_layout()
        plt.savefig(f"figures/pso_frame_iter_{h['iteration']:03d}.png", dpi=300)
        plt.close()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_curve, f(x_curve), linewidth=2, label="f(x)=x^2")
    target_zone = ax.axvspan(-threshold, threshold, alpha=0.15, label="Target zone |x| <= 1")
    particles_plot = ax.scatter([], [], s=50, label="Particles")
    gbest_plot = ax.scatter([], [], marker="x", s=120, label="Global best")
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 100)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.legend(loc="upper center", ncol=2, fontsize=8)

    def init():
        particles_plot.set_offsets(np.empty((0, 2)))
        gbest_plot.set_offsets(np.empty((0, 2)))
        return particles_plot, gbest_plot

    def animate(frame_index):
        h = history[frame_index]
        coords = np.column_stack((h["positions"], f(h["positions"])))
        particles_plot.set_offsets(coords)
        gbest_plot.set_offsets([[h["gbest_position"], f(h["gbest_position"])]])
        ax.set_title(
            f"1D PSO, w=0.7 | Iteration {h['iteration']} | Near optimum: {h['ratio_near']*100:.0f}%"
        )
        return particles_plot, gbest_plot

    selected_indices = list(range(0, len(history), 2))
    ani = FuncAnimation(
        fig,
        animate,
        frames=selected_indices,
        init_func=init,
        interval=120,
        blit=False,
    )
    ani.save("figures/pso_animation_w07.gif", writer=PillowWriter(fps=8))
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()