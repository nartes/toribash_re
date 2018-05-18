import numpy
import ctypes
import io
import enum
import os
import time
import sysv_ipc

class message_t(enum.Enum):
    TORIBASH_STATE          = 1
    TORIBASH_ACTION         = 2
    TORIBASH_LUA_DOSTRING   = 3

class toribash_world_state_t(ctypes.Structure):
    _fields_ = [('match_frame', ctypes.c_int32)]

class toribash_state_t(ctypes.Structure):
    class player_t(ctypes.Structure):
        _fields_ = [
            ('joints', ctypes.c_int32 * 20),
            ('grips', ctypes.c_int32 * 2),
            ('score', ctypes.c_double),
            ('injury', ctypes.c_double),
            ('joints_pos_3d', ctypes.c_double * 20 * 3),
            ]

    _pack_ = 1

    _fields_ = [
        ('players', player_t * 2),
        ('world_state', toribash_world_state_t)]

    def to_tensor(self):
        return numpy.concatenate([
            self.players[0].joints,
            self.players[0].grips,
            [self.players[0].score, self.players[0].injury],
            numpy.array(self.players[0].joints_pos_3d).reshape(-1),
            self.players[1].joints,
            self.players[1].grips,
            [self.players[1].score, self.players[1].injury],
            numpy.array(self.players[1].joints_pos_3d).reshape(-1),
            ])

    DIM = (20 + 2 + 2 + 20 * 3) * 2

class toribash_action_t(ctypes.Structure):
    class player_t(ctypes.Structure):
        _fields_ = [
            ('joints', ctypes.c_int32 * 20),
            ('grips', ctypes.c_int32 * 2)]
    _fields_ = [('players', player_t * 2)]

    DIM = (20 + 2) * 2
    BOUNDS = numpy.array(([[1,4],] * 20 + [[0,1],] * 2) * 2, dtype=numpy.int32)

    def to_tensor(self):
        return numpy.concatenate([
            self.players[0].joints,
            self.players[0].grips,
            self.players[1].joints,
            self.players[1].grips])

    @classmethod
    def from_tensor(cls, act_tensor):
        act = cls()

        for p in [0, 1]:
            for j in range(0, 20):
                act.players[p].joints[j] = act_tensor[p * 22 + j]
            for g in range(0, 2):
                act.players[p].grips[g] = act_tensor[p * 22 + g + 20]

        return act

class toribash_lua_dostring_t(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ('len', ctypes.c_uint32),
        ('buf', ctypes.c_char * 4096)]

class ToribashEnvironment:
    TORIBASH_MSG_QUEUE_KEY = ctypes.c_int32(0xffaaffbb)
    MAX_MESSAGE_SIZE = 8192

    def __init__(self):
        self._msg_queue = sysv_ipc.MessageQueue(
            self.TORIBASH_MSG_QUEUE_KEY.value,
            max_message_size = self.MAX_MESSAGE_SIZE)

    def read_state(self):
        message_buf, message_type = \
            self._msg_queue.receive(block=True, type=message_t.TORIBASH_STATE.value)

        assert message_type == message_t.TORIBASH_STATE.value

        st = toribash_state_t.from_buffer_copy(message_buf)

        return st

    def make_action(self, a_tensor):
        act = toribash_action_t.from_tensor(a_tensor)

        self._msg_queue.send(bytes(act), block=True, type=message_t.TORIBASH_ACTION.value)

    def lua_dostring(self, lua_str):
        lua_ds = toribash_lua_dostring_t()
        lua_ds.buf = lua_str
        lua_ds.len = len(lua_ds.buf)

        self._msg_queue.send(
            bytes(lua_ds),
            block=True,
            type=message_t.TORIBASH_LUA_DOSTRING.value)

class SimpleEnvironment:
    def __init__(self):
        self._edges = [
            [0, 1, -1],
            [0, 3, 1],
            [1, 2, -1],
            [1, 4, 1],
            [3, 2, -1],
            [3, 2, 1],
            [2, 5, -1],
            [2, 5, 1],
            [4, 5, -1],
            [4, 6, 1]]

        self._values = [
            [0, 0],
            [1, 0],
            [3, 3],
            [2, 1],
            [4, 2],
            [5, -4],
            [6, 1]
            ]

        self._s_dim = 1
        self._a_dim = 1
        self._a_bound = numpy.array([[0, 1]]).T

        self.reset()

    def get_parameters(self):
        return self._s_dim, self._a_dim, self._a_bound

    def reset(self):
        self._state = 0
        self._reward_sum = 0

    def read_state(self):
        return numpy.array([self._state], dtype=numpy.float32)

    def make_action(self, _action):
        action = 2 * numpy.int32(_action)[0] - 1

        a = numpy.array(self._edges)
        b = numpy.array(self._values)

        i = numpy.where(numpy.logical_and(
            a[:, 0] == self._state,
            a[:, 2] == action))[0]

        if len(i) == 0:
            raise ValueError("Invalid action")

        new_state = a[i[0], 1]

        i2 = numpy.where(b[:, 0] == new_state)[0]

        if len(i2) == 0:
            raise ValueError("Invalid new_state, no value")

        reward = b[i2[0], 1]

        self._reward_sum += reward

        self._state = new_state

        return self.read_state(), reward
