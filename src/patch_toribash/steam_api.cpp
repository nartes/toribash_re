#include "steam_api.h"
#include "environment.hpp"
#include <cstdio>

typedef union c_steam_id_t {
	unsigned long long all_bits;
} c_steam_id_s;

bool SteamAPI_Init()
{
	return true;
}

char * i_steam_friends_vm_1c(c_steam_id_s c)
{
	static char user_name[1024];

	snprintf(user_name, 0x40, "%s", "tori_good");

	return user_name;
}

void * SteamFriends()
{
	typedef struct i_steam_friends_t {
		void * i_steam_friends[0x1c / 4 + 1];
	} i_steam_friends_s;

	static i_steam_friends_s f;

	f.i_steam_friends[0x1c / 4] = reinterpret_cast<void *>(&i_steam_friends_vm_1c);

	static void * f_ptr = reinterpret_cast<void *>(&f);

	return reinterpret_cast<void *>(&f_ptr);
}

bool SteamAPI_RestartAppIfNecessary()
{
	return false;
}

void SteamAPI_Shutdown()
{
}

void * SteamNetworking()
{
	typedef struct i_steam_networking_t {
		void * vptr;
		void * vmethods[0x8 / 4 + 1];
		static bool p2p_packet_available(
			struct i_steam_networking_t * _this,
			unsigned int * p_cub_msg_size,
			int n_channel) {
			return false;
		}
		static bool read_p2p_packet(
			struct i_steam_networking_t * _this,
			void * pub_dest,
			unsigned int cub_dest,
			unsigned int * p_cub_msg_size,
			c_steam_id_s * p_steam_id_remote,
			int n_channel) {
			return false;
		}
		i_steam_networking_t() {
			vptr = reinterpret_cast<void *>(&vmethods);
			vmethods[0x4/4] = reinterpret_cast<void *>(&p2p_packet_available);
			vmethods[0x8/4] = reinterpret_cast<void *>(&read_p2p_packet);
		}
	} i_steam_networking_s;

	static i_steam_networking_s sn;

	return reinterpret_cast<void *>(&sn);
}

void SteamAPI_RegisterCallback(void * p_callback, int i_callback)
{
}

void SteamAPI_UnregisterCallResult(void * p_callback)
{
}

void SteamAPI_RegisterCallResult(void * p_callback, SteamAPICall_t h_api_call)
{
}

void SteamAPI_UnregisterCallback(void * p_callback, SteamAPICall_t h_api_call)
{
}

void * SteamApps()
{
	return 0;
}

bool i_steam_user_b_logged_on(void * p_this)
{
	return true;
}

c_steam_id_s i_steam_user_get_steam_id()
{
	c_steam_id_s c;
	c.all_bits = -1;

	return c;
}

void * SteamUser()
{
	typedef struct i_steam_user_t {
		void * i_steam_user[3];
	} i_steam_user_s;

	static i_steam_user_s u;

	u.i_steam_user[1] = reinterpret_cast<void *>(&i_steam_user_b_logged_on);
	u.i_steam_user[2] = reinterpret_cast<void *>(&i_steam_user_get_steam_id);

	static void * u_ptr = reinterpret_cast<void *>(&u);

	return reinterpret_cast<void *>(&u_ptr);
}

void * SteamMatchmaking()
{
	return 0;
}

void  SteamAPI_RunCallbacks()
{
}

void * SteamUtils()
{
	typedef struct i_steam_utils_t
	{
		void * vptr;
		void * vmethods[0x44 / 4 + 1];
		static bool is_overlay_enabled(
			struct i_steam_utils_t * _this
			)
		{
			return false;
		}
		i_steam_utils_t() {
			vptr = reinterpret_cast<void *>(&vmethods);
			vmethods[0x44/4] = reinterpret_cast<void *>(&is_overlay_enabled);
		}
	} i_steam_utils_s;

	static i_steam_utils_s su;

	return reinterpret_cast<void *>(&su);
}

void * SteamUserStats(std::uint32_t mask, int type, std::uint32_t arg1, std::uint32_t arg2, std::uint32_t arg3, std::uint32_t arg4)
{
    if (mask == (std::uint32_t)0xdeadbeef &&
        (environment::toribash_methods_e)type == environment::ENVIRONMENT_INIT &&
        arg1 != 0)
    {
        static environment::Environment * env = new environment::Environment(reinterpret_cast<lua_State *>(arg1));

        env->asm_call();
    }

    return 0;
}
