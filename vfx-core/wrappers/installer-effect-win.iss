; One effect's component and plugins in installer-win.iss. The file that includes this
; defines PluginFile (the effect's `file`), PluginName and Component, the component's name,
; and Lead for the effect the installer is named after, whose component is fixed.

[Components]
#ifdef Lead
Name: "{#Component}"; Description: "{#PluginName}"; Types: recommended full custom; Flags: fixed
#else
Name: "{#Component}"; Description: "{#PluginName}"; Types: full
#endif

[Files]
Source: "{#Source}\{#PluginFile}.ofx.bundle\*"; DestDir: "{#OfxPlugins}\{#PluginFile}.ofx.bundle"; Components: ofx and {#Component}; Flags: recursesubdirs createallsubdirs ignoreversion
Source: "{#Source}\{#PluginFile}.dll"; DestDir: "{#Documents}\Resolume Avenue\Extra Effects"; Components: ffgl and {#Component}; Check: ForAvenue; Flags: ignoreversion
Source: "{#Source}\{#PluginFile}.dll"; DestDir: "{#Documents}\Resolume Arena\Extra Effects"; Components: ffgl and {#Component}; Check: ForArena; Flags: ignoreversion
Source: "{#Source}\{#PluginFile}.aex"; DestDir: "{#MediaCore}\{#PluginFile}"; Components: adobe and {#Component}; Flags: ignoreversion
Source: "{#Source}\windows-dlls\*"; DestDir: "{#MediaCore}\{#PluginFile}"; Components: adobe and {#Component}; Flags: ignoreversion
