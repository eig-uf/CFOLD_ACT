"""Task progression only; no hardware or policy dependencies."""
from enum import Enum, auto
import time


class State(Enum):
    READY = auto()
    FOLD_1 = auto()
    VERIFY_1 = auto()
    FOLD_2 = auto()
    VERIFY_2 = auto()
    COMPLETE = auto()
    ABORTED = auto()


class StateMachine:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.state = State.READY
        self.started = None
        self.abort_reason = None
        self.verified = [False, False]

    @property
    def stage_index(self):
        if self.state in (State.FOLD_1, State.VERIFY_1):
            return 0
        if self.state in (State.FOLD_2, State.VERIFY_2):
            return 1
        return None

    def _transition(self, expected, target):
        if self.state not in expected:
            raise RuntimeError(f'Invalid transition: {self.state.name} -> {target.name}')
        self.state = target

    def start(self):
        self._transition((State.READY,), State.FOLD_1)
        self.started = self.clock()

    def execution_done(self):
        if self.state == State.FOLD_1:
            self._transition((State.FOLD_1,), State.VERIFY_1)
        else:
            self._transition((State.FOLD_2,), State.VERIFY_2)

    def verify(self, passed):
        if self.state not in (State.VERIFY_1, State.VERIFY_2):
            raise RuntimeError('Verification only allowed in verification states')
        i = self.stage_index
        if passed is False:
            self.abort(f'Fold {i + 1} verification failed')
            return
        self.verified[i] = passed is True
        self._transition((State.VERIFY_1, State.VERIFY_2), State.FOLD_2 if i == 0 else State.COMPLETE)
        self.started = self.clock() if i == 0 else None

    def check_timeout(self, timeout_s):
        if self.state in (State.FOLD_1, State.FOLD_2) and self.clock() - self.started >= timeout_s:
            self.abort('Stage timeout')
            raise TimeoutError('Stage timeout')

    def abort(self, reason):
        if self.state == State.ABORTED:
            return
        self._transition(tuple(s for s in State if s not in (State.COMPLETE, State.ABORTED)), State.ABORTED)
        self.abort_reason = str(reason)

    def reset(self):
        self._transition((State.COMPLETE, State.ABORTED), State.READY)
        self.started = None
        self.abort_reason = None
        self.verified = [False, False]
