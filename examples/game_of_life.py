import jax.numpy as jnp
from jax.lax import dynamic_slice
from jax.nn import softmax

from genjax import flip, gen, trace
from genjax import modular_vmap as vmap

neighbors_filter = jnp.array(
    [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
)


@gen  #########            (3, 3)      ()
def get_cell_from_window(window, flip_prob):
    """
    Given a 3x3 window, generate the value taken by the
    cell at the center of the window at the next
    time step.

    Args:
        window: A (3, 3) array of zeros and ones.
        flip_prob: The probability of flipping the cell
            value at the next time step.
    """
    neighbors = jnp.sum(window * neighbors_filter)
    grid = window[1, 1]
    # Apply the rules of the Game of Life
    deterministic_next_bit = jnp.where(
        (grid == 1) & ((neighbors == 2) | (neighbors == 3)),
        1,
        jnp.where((grid == 0) & (neighbors == 3), 1, 0),
    )
    p_is_one = jnp.where(deterministic_next_bit == 1, 1 - flip_prob, flip_prob)
    bit = trace("bit", flip, (p_is_one,))
    return bit


def get_windows(state):
    """
    Returns a (n_y, n_x, 3, 3) array of windows.
    For each cell in the given state, this array will
    contain a (3, 3) window centered at that cell.
    """
    n_y, n_x = state.shape
    # pad the state with zeros
    state_padded = jnp.pad(state, 1)
    # windows of shape (n_y, n_x, 3, 3)
    return vmap(
        lambda i: vmap(lambda j: dynamic_slice(state_padded, (i, j), (3, 3)))(
            jnp.arange(n_x)
        )
    )(jnp.arange(n_y))


@gen
def generate_next_state(prev_state, p_flip):
    """
    Given a game of life state, generate the next state.
    """
    # windows of shape (n_y, n_x, 3, 3)
    windows = get_windows(prev_state)
    # construct next state; shape is (n_y, n_x)
    return trace(
        "cells",
        get_cell_from_window.vmap(
            in_axes=(0, None)  # map over axis 1
        ).vmap(
            in_axes=(0, None)  # map over axis 0
        ),
        (windows, p_flip),
    )


@gen
def generate_state_pair(n_y, n_x, p_flip):
    """
    Generate an initial game of life state uniformly at random,
    and generate the next state given the initial state.

    Args:
        n_y: The number of rows in the state. (Const)
        n_x: The number of columns in the state. (Const)
        p_flip: The probability of flipping a cell value.
    """
    init = trace("init", flip.vmap().vmap(), (0.5 * jnp.ones((n_y, n_x)),))
    step = trace("step", generate_next_state, (init, p_flip))
    return (init, step)


### Single Cell Gibbs Update Implementation ###

AND = jnp.logical_and


def get_gibbs_probs_fast(i, j, current_state, future_state, p_flip):
    relevant_next_ys = jnp.array([i - 1, i, i + 1])
    relevant_next_xs = jnp.array([j - 1, j, j + 1])
    current_state_padded = jnp.pad(current_state, 1)

    def window_for_next_cell_at_offset(next_y, next_x, val_at_ij):
        window = dynamic_slice(current_state_padded, (next_y, next_x), (3, 3))
        a, b = i - next_y + 1, j - next_x + 1
        is_in_range = AND(AND(0 <= a, a < 3), AND(0 <= b, b < 3))
        val = jnp.where(is_in_range, val_at_ij, window[a, b])
        window = window.at[a, b].set(val)
        return window

    def get_score_for_val(val_at_ij):
        scores = vmap(
            lambda next_y: vmap(
                lambda next_x: get_cell_from_window.assess(
                    (window_for_next_cell_at_offset(next_y, next_x, val_at_ij), p_flip),
                    {"bit": future_state[next_y, next_x]},
                )[0]
            )(relevant_next_xs)
        )(relevant_next_ys)
        mask = vmap(
            lambda next_y: vmap(
                lambda next_x: AND(
                    AND(0 <= next_y, next_y < future_state.shape[1]),
                    AND(0 <= next_x, next_x < future_state.shape[0]),
                )
            )(relevant_next_xs)
        )(relevant_next_ys)
        return jnp.sum(jnp.where(mask, scores, 0))

    scores = vmap(get_score_for_val)(jnp.array([0, 1]))

    return softmax(scores)


@gen
def gibbs_move_on_cell_fast(i, j, current_state, future_state, p_flip):
    """Gibbs sample a bit for the position (i, j). Runs in O(1)."""
    p_zero = get_gibbs_probs_fast(i, j, current_state, future_state, p_flip)[0]
    val = trace("bit", flip, (1 - p_zero,))
    return val


## Version with Gibbs sampling ##
def get_scores_using_gfi(i, j, current_state, future_state, p_flip):
    current_trace, _ = generate_state_pair.importance(
        C["init", :].set(current_state)
        ^ C["step", "cells", :, :, "bit"].set(future_state),
        (current_state.shape[0], current_state.shape[1], p_flip),
    )
    _, wt0, _, _ = current_trace.update(C["init", i, j].set(0))
    _, wt1, _, _ = current_trace.update(C["init", i, j].set(1))
    return softmax(jnp.array([wt0, wt1]))


@gen
def gibbs_move_on_cell_using_gfi(i, j, current_state, future_state, p_flip):
    """
    Gibbs sample a bit for the position (i, j).
    Runs in O(n cells in board).
    """
    p_zero = get_scores_using_gfi(i, j, current_state, future_state, p_flip)[0]
    return trace("bit", flip, (1 - p_zero,))


### Full Gibbs Sweep Implementation ###


@gen
def gibbs_move_on_all_cells_at_offset(
    oy: int,
    ox: int,
    current_state,
    future_state,
    p_flip,
):
    i_vals = jnp.arange(oy, current_state.shape[0], 3)
    j_vals = jnp.arange(ox, current_state.shape[1], 3)
    vals = trace(
        "cells",
        gibbs_move_on_cell_fast.vmap(in_axes=(None, 0, None, None, None)).vmap(
            in_axes=(0, None, None, None, None)
        ),
        (i_vals, j_vals, current_state, future_state, p_flip),
    )
    return current_state.at[*jnp.ix_(i_vals, j_vals)].set(vals)


def full_gibbs_sweep(
    current_state,
    future_state,
    p_flip,
):
    for oy in range(3):
        for ox in range(3):
            current_state = gibbs_move_on_all_cells_at_offset(
                oy,
                ox,
                current_state,
                future_state,
                p_flip,
            )
    return current_state
