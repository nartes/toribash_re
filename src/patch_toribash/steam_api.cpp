#include "steam_api.h"

bool SteamAPI_Init()
{
	return true;
}

void * SteamFriends()
{
	return 0;
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
	return 0;
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

typedef union c_steam_id_t {
	unsigned long long all_bits;
} c_steam_id_s;


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
	return 0;
}

void * SteamUserStats()
{
	return 0;
}
