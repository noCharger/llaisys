target("llaisys-device-nvidia")
    set_kind("static")
    add_rules("cuda")
    set_languages("cxx17")
    
    if is_plat("linux") then
        add_cugencodes("native")
        add_cuflags("-Xcompiler -fPIC")
    end

    add_files("../src/device/nvidia/*.cu")
target_end()

target("llaisys-ops-nvidia")
    set_kind("static")
    add_rules("cuda")
    set_languages("cxx17")
    add_deps("llaisys-tensor")

    if is_plat("linux") then
        add_cugencodes("native")
        add_cuflags("-Xcompiler -fPIC")
    end

    add_links("cublas")
    add_files("../src/ops/*/nvidia/*.cu")
target_end()
