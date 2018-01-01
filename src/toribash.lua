local function _end_game()
    print("finish_game")

    local h = (47 * math.random(1000000000) + math.random(1000000000))
    local n = "replay_"..(h).."_"..(os.time())
    run_cmd("sr "..n)

    start_new_game()
end

local function _random_change()
    for p = 0,1 do
        for n, i in pairs(JOINTS) do
            set_joint_state(p, i, math.random(4))
        end
        set_grip_info(p, BODYPARTS.L_HAND, math.random(2) - 1)
        set_grip_info(p, BODYPARTS.R_HAND, math.random(2) - 1)
    end
end

local function _enter_freeze()
    _random_change()
    step_game()
end

local function _new_game()
    print("start_game")

    _enter_freeze()
end

function _toggle_ui()
      local v = get_option("tori")

      v = (v + 1) % 2
      set_option("tori", v)
      set_option("uke", v)
      set_option("hud", v)
end

function main()
    print("init toribash.lua")

    add_hook("enter_freeze", "toribash", _enter_freeze)
    add_hook("end_game", "toribash", _end_game)
    add_hook("new_game", "toribash", _new_game)
end
