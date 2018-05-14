#pragma once

extern "C" {
#include <deps/lua/lua.h>
#include <deps/lua/lauxlib.h>
};

#include <cstdio>
#include <string>
#include <cstdint>
#include <vector>
#include <stdexcept>
#include <sys/msg.h>
#include <cstring>

namespace environment {

typedef enum message_t
{
    TORIBASH_STATE          = 1,
    TORIBASH_ACTION         = 2,
    TORIBASH_LUA_DOSTRING   = 3,
} message_e;

typedef struct toribash_state_t
{
    struct player {
        int joints[20];
        int grips[2];
        double score;
    } players[2];
} toribash_state_s;

typedef struct toribash_action_t
{
    struct player {
        int joints[20];
        int grips[2];
    } players[2];
} toribash_action_s;

struct toribash_lua_dostring_t
{
    std::uint32_t len;
    char buf[4096];
};

typedef enum toribash_methods_t
{
    ENVIRONMENT_INIT    = 0x5
} toribash_methods_e;

class Environment
{

public:
    Environment(lua_State * lua_state);

    void asm_call();

private:
    void _dump_state();
    void _parse_action();
    void _lua_dostring();

    template<typename T>
    void send_message(message_t message_type, const T * src);

    template<typename T>
    T * recv_message(message_t message_type);

private:
    lua_State * _lua_state;

    static Environment * _env;

    int _msg_queue_id;
    const int TORIBASH_MSG_QUEUE_KEY = 0xffaaffbb;


    std::vector<std::uint8_t> out_buffer;
    std::vector<std::uint8_t> in_buffer;
};


template<typename T>
void Environment::send_message(message_t message_type, const T * src)
{
    out_buffer.resize(sizeof(long) + sizeof(T));

    auto mb1 = reinterpret_cast<msgbuf *>(out_buffer.data());

    mb1->mtype = (int)message_type;

    memcpy(out_buffer.data() + sizeof(long), src, sizeof(T));

    int ret = msgsnd(_msg_queue_id, mb1, sizeof(T), 0);

    if (ret == -1)
    {
        throw std::runtime_error("msgsnd has failed");
    }
}

template<typename T>
T * Environment::recv_message(message_t message_type)
{
    in_buffer.resize(sizeof(long) + sizeof(T));

    auto mb1 = reinterpret_cast<msgbuf *>(in_buffer.data());

    int ret = msgrcv(_msg_queue_id, mb1, sizeof(T), message_type, 0);

    if (ret != sizeof(T) || mb1->mtype != (int)message_type)
    {
        printf("ret = %d, mb1->mtype = %d, sizeof(T) = %d\n", ret, mb1->mtype, (int)sizeof(T));
        throw std::runtime_error("msgrcv has failed");
    }

    return reinterpret_cast<T *>(in_buffer.data() + sizeof(long));
}

};
