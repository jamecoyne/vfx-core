; One effect's plugins in installer-win.iss. The file that includes this defines PluginFile
; (the effect's `file`), and Component when the effect is an option; the lead effect has no
; component and installs for every app chosen.

#ifdef Component
  #define For(str App) App + " and " + Component
[Components]
Name: "{#Component}"; Description: "{#PluginName}"
#else
  #define For(str App) App
#endif

[Files]
Source: "{#Source}\{#PluginFile}.ofx.bundle\*"; DestDir: "{#OfxPlugins}\{#PluginFile}.ofx.bundle"; Excludes: "dxcompiler.dll,DXC-*.txt"; Components: {#For("ofx")}; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Source}\{#PluginFile}.dll"; DestDir: "{#Arena}"; Components: {#For("arena")}; Flags: ignoreversion
Source: "{#Source}\{#PluginFile}.dll"; DestDir: "{#Avenue}"; Components: {#For("avenue")}; Flags: ignoreversion
Source: "{#Source}\{#PluginFile}.aex"; DestDir: "{#MediaCore}\{#PluginFile}"; Components: {#For("adobe")}; Flags: ignoreversion

; v0.1.1 put the compiler beside each OFX and After Effects plugin; the shared one replaces it.
[InstallDelete]
Type: files; Name: "{#OfxPlugins}\{#PluginFile}.ofx.bundle\Contents\Win64\dxcompiler.dll"; Components: {#For("ofx")}
Type: files; Name: "{#OfxPlugins}\{#PluginFile}.ofx.bundle\Contents\Win64\DXC-*.txt"; Components: {#For("ofx")}
Type: files; Name: "{#MediaCore}\{#PluginFile}\dxcompiler.dll"; Components: {#For("adobe")}
Type: files; Name: "{#MediaCore}\{#PluginFile}\DXC-*.txt"; Components: {#For("adobe")}

#undef For
