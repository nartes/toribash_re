function _toggle_ui()
      local v = get_option("tori")

      v = (v + 1) % 2
      set_option("tori", v)
      set_option("uke", v)
      set_option("hud", v)
end

function main()
    print("init toribash.lua")

    open_url()
    add_hook("enter_freeze", "patch_toribash", patch_toribash_enter_freeze_cb)
    add_hook("end_game", "patch_toribash", patch_toribash_end_game_cb)
    add_hook("new_game", "patch_toribash", patch_toribash_start_game_cb)
end

function read_socket()
    local i = 0
    local lines = {}
    local f = io.open("toribash_socket", "r")
    if f ~= nil then
        while f:read(0) ~= nil do
            f:read("*l")
            table.insert(lines, i + 1, line)
            i = i + 1
        end
        f:close()
    else
        print("Can't open toribash_socket")
    end

    return lines
end

dofile("dbg.lua")

main()
