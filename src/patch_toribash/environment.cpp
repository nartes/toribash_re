#include "environment.hpp"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>
#include <functional>
#include <unistd.h>
#include <ctime>
#include <sys/stat.h>

namespace environment
{


Environment* Environment::_env = 0;

Environment::Environment(lua_State * lua_state)
    : _lua_state(lua_state)
{
    _env = this;

    while ((_msg_queue_id = msgget(
        TORIBASH_MSG_QUEUE_KEY,
        IPC_CREAT | IPC_EXCL | S_IRUSR | S_IWUSR)) == -1)
    {
        _msg_queue_id = msgget(TORIBASH_MSG_QUEUE_KEY, 0);

        if (_msg_queue_id == -1 || msgctl(_msg_queue_id, IPC_RMID, 0) == -1)
        {
            throw std::runtime_error("msgget has failed");
        }
    }
}

void Environment::_parse_action()
{
    auto act = recv_message<toribash_action_t>(TORIBASH_ACTION);

    for (int p = 0; p < 2; ++p)
    {
        lua_getglobal(_lua_state, "JOINTS");
        int t = lua_gettop(_lua_state);

        lua_pushnil(_lua_state);
        while (lua_next(_lua_state, t) != 0)
        {
            int k = lua_tonumber(_lua_state, -2);
            int v = lua_tonumber(_lua_state, -1);
            lua_pop(_lua_state, 1);


            lua_getglobal(_lua_state, "set_joint_state");
            lua_pushnumber(_lua_state, p);
            lua_pushnumber(_lua_state, v);
            lua_pushnumber(_lua_state, act->players[p].joints[v]);
            lua_call(_lua_state, 3, 0);
        }

        int i = 0;

        for (int hand_id : std::vector<int>({11, 12}))
        {
            lua_getglobal(_lua_state, "set_grip_info");
            lua_pushnumber(_lua_state, p);
            lua_pushnumber(_lua_state, hand_id);
            lua_pushnumber(_lua_state, act->players[p].grips[i++]);
            lua_call(_lua_state, 3, 0);
        }
    }
}

void Environment::_lua_dostring()
{
    for (
        auto lua_ds = recv_message<toribash_lua_dostring_t>(TORIBASH_LUA_DOSTRING);
        lua_ds->len != 0;
        lua_ds = recv_message<toribash_lua_dostring_t>(TORIBASH_LUA_DOSTRING)
        )
    {
        lua_ds->buf[sizeof(lua_ds->buf) - 1] = 0;
        luaL_loadstring(_lua_state, lua_ds->buf);
        lua_pcall(_lua_state, 0, LUA_MULTRET, 0);
    }
}

void Environment::_dump_state()
{
    toribash_state_s st;

    for (int p = 0; p < 2; ++p)
    {
        lua_getglobal(_lua_state, "JOINTS");
        int t = lua_gettop(_lua_state);

        lua_pushnil(_lua_state);
        while (lua_next(_lua_state, t) != 0)
        {
            int k = lua_tonumber(_lua_state, -2);
            int v = lua_tonumber(_lua_state, -1);
            lua_pop(_lua_state, 1);

            lua_getglobal(_lua_state, "get_joint_info");
            lua_pushnumber(_lua_state, p);
            lua_pushnumber(_lua_state, v);
            lua_call(_lua_state, 2, 1);

            lua_getfield(_lua_state, lua_gettop(_lua_state), "state");

            st.players[p].joints[v] = lua_tonumber(_lua_state, -1);

            lua_pop(_lua_state, 2);

            lua_getglobal(_lua_state, "get_joint_pos");
            lua_pushnumber(_lua_state, p);
            lua_pushnumber(_lua_state, v);
            lua_call(_lua_state, 2, 3);

            st.players[p].joints_pos_3d[v][0] = lua_tonumber(_lua_state, -3);
            st.players[p].joints_pos_3d[v][1] = lua_tonumber(_lua_state, -2);
            st.players[p].joints_pos_3d[v][2] = lua_tonumber(_lua_state, -1);

            lua_pop(_lua_state, 3);
        }

        int i = 0;

        for (int hand_id : std::vector<int>({11, 12}))
        {
            lua_getglobal(_lua_state, "get_grip_info");
            lua_pushnumber(_lua_state, p);
            lua_pushnumber(_lua_state, hand_id);
            lua_call(_lua_state, 2, 1);

            st.players[p].grips[i++] = lua_tonumber(_lua_state, -1);

            lua_pop(_lua_state, 1);
        }

        lua_getglobal(_lua_state, "get_player_info");
        lua_pushnumber(_lua_state, p);
        lua_call(_lua_state, 1, 1);

        lua_getfield(_lua_state, lua_gettop(_lua_state), "score");

        st.players[p].score = lua_tonumber(_lua_state, -1);

        lua_pop(_lua_state, 1);

        lua_getfield(_lua_state, lua_gettop(_lua_state), "injury");

        st.players[p].injury = lua_tonumber(_lua_state, -1);

        lua_pop(_lua_state, 2);
    }

    lua_getglobal(_lua_state, "get_world_state");
    lua_call(_lua_state, 0, 1);

    lua_getfield(_lua_state, lua_gettop(_lua_state), "match_frame");

    st.world_state.match_frame = lua_tointeger(_lua_state, -1);

    lua_pop(_lua_state, 2);

    send_message(TORIBASH_STATE, &st);
}

void Environment::asm_call()
{
    int n = lua_gettop(_lua_state);

    for (int i = 1; i <= n; ++i)
    {
        lua_pushvalue(_lua_state, i);

        const char * str = lua_tostring(_lua_state, n + 1);

        printf("[%d], type = %d, str = %s\n",
            i, lua_type(_lua_state, i),
            str ? str : "");

        if (str)
        {
            lua_settop(_lua_state, n);
        }
    }

    static auto enter_freeze_cb = [] (lua_State * L)
    {
        printf("enter_freeze is called\n");

        Environment::_env->_lua_dostring();

        Environment::_env->_dump_state();

        Environment::_env->_parse_action();

        lua_getglobal(L, "step_game");
        lua_call(L, 0, 0);

        return (int)0;
    };

    static auto end_game_cb = [] (lua_State * L)
    {
        printf("end_game is called\n");

        lua_getglobal(L, "start_new_game");
        lua_call(L, 0, 0);

        return (int)0;
    };

    static auto start_game_cb = [] (lua_State * L)
    {
        printf("start_game is called\n");

        lua_getglobal(L, "patch_toribash_enter_freeze_cb");
        lua_call(L, 0, 0);

        return (int)0;
    };

    //lua_getglobal(_lua_state, "add_hook");
    //lua_pushstring(_lua_state, "enter_freeze");
    //lua_pushstring(_lua_state, "patch_toribash");
    lua_pushcfunction(_lua_state, static_cast<int (*)(lua_State *)>(enter_freeze_cb));
    lua_setglobal(_lua_state, "patch_toribash_enter_freeze_cb");
    lua_pushcfunction(_lua_state, start_game_cb);
    lua_setglobal(_lua_state, "patch_toribash_start_game_cb");
    lua_pushcfunction(_lua_state, end_game_cb);
    lua_setglobal(_lua_state, "patch_toribash_end_game_cb");
    //lua_call(_lua_state, 2, 1);

    /*
    const char * str = lua_tostring(_lua_state, lua_gettop(_lua_state));

    printf("[%d], type = %d, str = %s\n",
        lua_gettop(_lua_state), lua_type(_lua_state, lua_gettop(_lua_state)),
        str ? str : "");

    if (str)
    {
        lua_settop(_lua_state, lua_gettop(_lua_state) - 1);
    }
    */
}

};
