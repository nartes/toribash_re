#pragma once

extern "C" {
#include <deps/lua/lua.h>
};

#include <cstdio>

namespace environment {

typedef enum message_t
{
    TORIBASH_STATE      = 0,
    TORIBASH_ACTION     = 1
} message_e;

typedef struct toribash_state_t
{
    struct player {
        int joints[20];
        int grips[2];
    } players[2];
} toribash_state_s;

typedef struct toribash_action_t
{
    struct player {
        int joints[20];
        int grips[2];
    } players[2];
} toribash_action_s;

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

private:
    lua_State * _lua_state;

    static Environment * _env;
};

};
