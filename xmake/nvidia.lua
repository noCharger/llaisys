target("llaisys-device-nvidia")
    set_kind("object")
    add_rules("cuda")
    set_languages("cxx17")

    -- Enable device linking globally for this target
    set_policy("build.cuda.devlink", true)
    
    -- Link CUDA runtime and driver
    add_links("cudart", "cublas", "cuda")

    if is_plat("linux") then
        add_cugencodes("native")
        add_cuflags("-Xcompiler -fPIC")
        add_values("cuda.rdc", true)
        set_toolchains("cuda")
    end

    add_files("../src/device/nvidia/*.cu")

    -- Safely locate nvcc and append the lib64/stubs directory
    on_load(function (target)
        import("lib.detect.find_tool")
        local nvcc = find_tool("nvcc")
        if nvcc and nvcc.program then
            local nvcc_path = nvcc.program
            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
        end
    end)
target_end()

target("llaisys-ops-nvidia")
    set_kind("object")
    add_rules("cuda")
    set_languages("cxx17")
    add_deps("llaisys-tensor")

    -- Enable device linking globally for this target
    set_policy("build.cuda.devlink", true)

    -- Link CUDA runtime and driver (replacing the standalone "cublas" link)
    add_links("cudart", "cublas", "cuda", "curand")

    if is_plat("linux") then
        add_cugencodes("native")
        add_cuflags("-Xcompiler -fPIC")
        add_values("cuda.rdc", true)
        set_toolchains("cuda")
    end

    add_files("../src/ops/*/nvidia/*.cu")

    -- Safely locate nvcc and append the lib64/stubs directory
    on_load(function (target)
        import("lib.detect.find_tool")
        local nvcc = find_tool("nvcc")
        if nvcc and nvcc.program then
            local nvcc_path = nvcc.program
            target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs")
        end
    end)
target_end()