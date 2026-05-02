add_rules("mode.debug", "mode.release")
set_encodings("utf-8")

add_includedirs("include")
add_includedirs("src")

-- CPU --
includes("xmake/cpu.lua")

-- NVIDIA --
option("nv-gpu")
    set_default(false)
    set_showmenu(true)
    set_description("Whether to compile implementations for Nvidia GPU")
option_end()

if has_config("nv-gpu") then
    add_defines("ENABLE_NVIDIA_API")
    includes("xmake/nvidia.lua")
end

target("llaisys-utils")
    set_kind("static")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/utils/*.cpp")

    on_install(function (target) end)
target_end()


target("llaisys-device")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-device-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/device/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-core")
    set_kind("static")
    add_deps("llaisys-utils")
    add_deps("llaisys-device")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/core/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-tensor")
    set_kind("static")
    add_deps("llaisys-core")

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end

    add_files("src/tensor/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys-ops")
    set_kind("static")
    add_deps("llaisys-ops-cpu")
    if has_config("nv-gpu") then
        add_deps("llaisys-ops-nvidia")
    end

    set_languages("cxx17")
    set_warnings("all", "error")
    if not is_plat("windows") then
        add_cxflags("-fPIC", "-Wno-unknown-pragmas")
    end
    
    add_files("src/ops/*/*.cpp")

    on_install(function (target) end)
target_end()

target("llaisys")
    set_kind("shared")
    
    if has_config("nv-gpu") then
        if is_plat("linux") then
            add_deps("llaisys-ops-nvidia")
            add_deps("llaisys-device-nvidia")
            add_links("cublas")
        end
        add_rules("cuda")
        
        -- Enable device linking for the final shared library link phase
        set_policy("build.cuda.devlink", true)
        
        -- force enable -fPIC
        if is_plat("linux") then
            add_cxflags("-fPIC")
        end
        
        -- Link cudadevrt if using RDC (often required implicitly by xmake/cuda rule when RDC is on)
        add_links("cudadevrt")
        
        -- Ensure RDC is enabled for the shared target as well
        if is_plat("linux") then
             add_values("cuda.rdc", true)
        end

        -- Ensure the final linker can find the cuda driver stubs
        on_load(function (target)
            import("lib.detect.find_tool")
            local nvcc = find_tool("nvcc")
            if nvcc ~= nil then 
                if is_plat("windows") then 
                    nvcc_path = os.iorun("where nvcc"):match("(.-)\r?\n") 
                else 
                    nvcc_path = nvcc.program 
                end 
    
                target:add("linkdirs", path.directory(path.directory(nvcc_path)) .. "/lib64/stubs") 
                target:add("links", "cuda") 
            end 
        end)
    end
    
    add_deps("llaisys-utils")
    add_deps("llaisys-device")
    add_deps("llaisys-core")
    add_deps("llaisys-tensor")
    add_deps("llaisys-ops")

    set_languages("cxx17")
    set_warnings("all", "error")
    add_files("src/llaisys/*.cc")

    if has_config("nv-gpu") then
        -- Glob picks up cuda_dummy.cu plus the paged-KV CUDA kernels
        -- (scatter, paged attention helpers).
        add_files("src/llaisys/*.cu")
    end

    set_installdir(".")

    after_install(function (target)
        -- copy shared library to python package
        print("Copying llaisys to python/llaisys/libllaisys/ ..")
        if is_plat("windows") then
            os.cp("bin/*.dll", "python/llaisys/libllaisys/")
        elseif is_plat("linux") then
            os.cp("lib/*.so", "python/llaisys/libllaisys/")
        elseif is_plat("macosx") then
            os.cp("lib/*.dylib", "python/llaisys/libllaisys/")
        end
    end)
target_end()