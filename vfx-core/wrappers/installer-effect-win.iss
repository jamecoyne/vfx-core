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
Source: "{#Source}\{#PluginFile}.ofx.bundle\*"; DestDir: "{#OfxPlugins}\{#PluginFile}.ofx.bundle"; Components: {#For("ofx")}; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Source}\{#PluginFile}.dll"; DestDir: "{#Arena}"; Components: {#For("arena")}; Flags: ignoreversion
Source: "{#Source}\{#PluginFile}.dll"; DestDir: "{#Avenue}"; Components: {#For("avenue")}; Flags: ignoreversion
Source: "{#Source}\{#PluginFile}.aex"; DestDir: "{#MediaCore}\{#PluginFile}"; Components: {#For("adobe")}; Flags: ignoreversion
Source: "{#Source}\windows-dlls\*"; DestDir: "{#MediaCore}\{#PluginFile}"; Components: {#For("adobe")}; Flags: ignoreversion

#undef For
