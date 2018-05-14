import numpy
import ctypes
import io
import enum
import os
import time

class message_e(enum.Enum):
    TORIBASH_STATE      = 0
    TORIBASH_ACTION     = 1

class toribash_state(ctypes.Structure):
    class player(ctypes.Structure):
        _fields_ = [
            ('joints', ctypes.c_int32 * 20),
            ('grips', ctypes.c_int32 * 2)]

    _fields_ = [('players', player * 2)]

    def to_tensor(self):
        return numpy.concatenate([
            self.players[0].joints,
            self.players[0].grips,
            self.players[1].joints,
            self.players[1].grips])

    DIM = (20 + 2) * 2

class toribash_action(ctypes.Structure):
    class player(ctypes.Structure):
        _fields_ = [
            ('joints', ctypes.c_int32 * 20),
            ('grips', ctypes.c_int32 * 2)]
    _fields_ = [('players', player * 2)]

    DIM = (20 + 2) * 2
    BOUNDS = numpy.array(([[1,4],] * 20 + [[1,2],] * 2) * 2, dtype=numpy.int32)

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

class ToribashEnvironment:
    def __init__(self):
        self._ddpg_socket_out = "/tmp/patch_toribash_environment_ddpg_socket_in"
        self._ddpg_socket_in =  "/tmp/patch_toribash_environment_ddpg_socket_out"

    def send_bytes(self, data):
        while os.path.exists(self._ddpg_socket_out) or \
            os.path.exists(self._ddpg_socket_out + '.lock'):
            time.sleep(0.001)

        io.open(self._ddpg_socket_out + '.lock', 'a').close()

        ddpg_socket_out = io.open(self._ddpg_socket_out, "wb")
        ddpg_socket_out.write(data)
        ddpg_socket_out.close()

        os.remove(self._ddpg_socket_out + '.lock')

    def read_bytes(self):
        while os.path.exists(self._ddpg_socket_in + '.lock') or \
            not os.path.exists(self._ddpg_socket_in):
           time.sleep(0.001)

        io.open(self._ddpg_socket_in + '.lock', 'a').close()

        while not os.path.exists(self._ddpg_socket_in + '.lock'):
            time.sleep(0.001)

        ddpg_socket_in = io.open(self._ddpg_socket_in, "rb")

        res =  ddpg_socket_in.read()
        ddpg_socket_in.close()

        os.remove(self._ddpg_socket_in)
        os.remove(self._ddpg_socket_in + '.lock')

        return res

    def read_state(self):
        z = self.read_bytes()

        assert ctypes.c_int32.from_buffer_copy(z[:4]).value == message_e.TORIBASH_STATE.value
        assert ctypes.sizeof(toribash_state) == ctypes.c_int32.from_buffer_copy(z[4:][:4]).value
        st = toribash_state.from_buffer_copy(z[8:])

        return st.to_tensor()

    def make_action(self, a_tensor):
        act = toribash_action.from_tensor(a_tensor)

        self.send_bytes(
            bytes(ctypes.c_int32(message_e.TORIBASH_ACTION.value)) +
            bytes(ctypes.c_int32(ctypes.sizeof(toribash_action))) +
            bytes(act))
