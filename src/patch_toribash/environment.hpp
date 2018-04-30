#pragma once

extern "C" {
#include <deps/lua/lua.h>
};

namespace environment {

typedef enum toribash_methods_t
{
    LUA_CALL            = 5,
    ENVIRONMENT_INIT    = 0x20
} toribash_methods_e;

extern "C" {
typedef void (* toribash_lua_pushcclosure_t)(lua_State *L, lua_CFunction fn, int);
};

class Environment
{
public:
    static void create_environment();

public:
    static lua_State * lua_state;

private:
    Environment();

private:
    static Environment * global_environment;
};

};
