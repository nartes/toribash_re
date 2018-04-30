#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool SteamAPI_Init();
void * SteamFriends();
bool SteamAPI_RestartAppIfNecessary();
void SteamAPI_Shutdown();
void * SteamNetworking();

typedef unsigned long long SteamAPICall_t;

void SteamAPI_RegisterCallback(void * p_callback, int i_callback);
void SteamAPI_UnregisterCallResult(void * p_callback);
void SteamAPI_RegisterCallResult(void * p_callback, SteamAPICall_t h_api_call);
void SteamAPI_UnregisterCallback(void * p_callback, SteamAPICall_t h_api_call);
void * SteamApps();
void * SteamUser();
void * SteamMatchmaking();
void SteamAPI_RunCallbacks();
void * SteamUtils();
void * SteamUserStats(std::uint32_t mask, int type, std::uint32_t arg1, std::uint32_t arg2, std::uint32_t arg3, std::uint32_t arg4);

#ifdef __cplusplus
}
#endif
