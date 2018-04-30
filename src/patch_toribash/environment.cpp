#include "environment.hpp"

namespace environment
{

Environment * Environment::global_environment = 0;

lua_State * Environment::lua_state = 0;

void Environment::create_environment()
{
    if (!global_environment)
    {
        global_environment = new Environment();
    }
}

Environment::Environment()
{
}

};
