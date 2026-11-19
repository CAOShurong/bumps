import asyncio
from copy import deepcopy

import h5py
import numpy as np
import pytest

from . import api
from .curve import Curve
from .fitproblem import FitProblem
from .fitters import DEFit, DreamFit
from .state import State


def _line(x, slope, offset):
    return slope * x + offset


def _linear_problem():
    x = np.linspace(0, 1, 11)
    curve = Curve(_line, x, 2 * x + 1, dy=0.1 * np.ones_like(x), slope=0.5, offset=0.5)
    curve.slope.range(-10, 10)
    curve.offset.range(-10, 10)
    return FitProblem(curve)


def test_de_resume_progress(monkeypatch):
    state = State()
    state.shared.selected_fitter = "de"
    state.shared.fitter_settings = deepcopy(api.state.shared.fitter_settings)
    state.shared.autosave_session = state.shared.autosave_history = False
    state.parallel = 1
    monkeypatch.setattr(api, "state", state)

    async def fit(steps, resume=False):
        await api.start_fit_thread("de", {"steps": steps, "pop": 4, "ftol": 0, "xtol": 0}, resume=resume)
        progress = state.shared.active_fit
        interval = (progress["step"], progress["num_steps"])
        await asyncio.wait_for(api.wait_for_fit_complete(), timeout=10)
        assert state.shared.active_fit == {}
        return interval, state.fitting.fit_state["step"][-1]

    async def check():
        try:
            await api.set_problem(_linear_problem())
            assert await fit(5) == ((0, 5), 5)
            assert await fit(3, resume=True) == ((5, 8), 8)
            state.fitting.method = "dream"
            assert await fit(3, resume=True) == ((0, 3), 3)
        finally:
            if state.fit_thread is not None:
                state.fit_abort_event.set()
                await asyncio.wait_for(api.wait_for_fit_complete(), timeout=10)

    rng_state = np.random.get_state()
    np.random.seed(267)
    try:
        asyncio.run(check())
    finally:
        np.random.set_state(rng_state)


def test_progress_steps_accepts_saved_de_history_and_preserves_dream(tmp_path):
    path = tmp_path / "fit.h5"
    with h5py.File(path, "w") as file:
        DEFit.h5dump(file, {"step": [2, 5]})
    with h5py.File(path, "r") as file:
        saved = DEFit.h5load(file)
    assert DEFit.progress_steps(options={"steps": 3}, fit_state=saved) == (5, 8)
    assert saved == {"step": [2, 5]}
    with pytest.raises(ValueError, match="saved step"):
        DEFit.progress_steps(options={"steps": 3}, fit_state={"step": []})

    problem = _linear_problem()
    options = {"steps": 3, "burn": 2}
    assert DreamFit.progress_steps(problem, options, fit_state=object()) == (
        0,
        DreamFit.max_steps(problem, options),
    )
