function saveFigure(fig, file, varargin)
%SAVEFIGURE  Save a figure to a PNG file without stopping the run.
%
%   saveFigure(fig, file, Name=Value...)   same arguments as exportgraphics
%
%   Tries exportgraphics first, then print. If both fail (for example
%   because the window was closed or the screen went to sleep), it prints
%   a warning and the run continues, so a figure problem can never stop
%   the project. Hidden figures (used by RUN_ME) are closed after saving.

if isempty(fig) || ~isgraphics(fig)
    warning("rfproject:noFigure", "Figure for %s no longer exists; not saved.", file);
    return
end
drawnow;
saved = false;
try
    exportgraphics(fig, file, varargin{:});
    saved = true;
catch err1
    try
        print(fig, file, "-dpng", "-r150");
        saved = true;
    catch err2
        warning("rfproject:saveFailed", "Could not save %s (%s | %s). Continuing.", ...
            file, err1.message, err2.message);
    end
end
if saved && isgraphics(fig) && strcmp(get(fig, "Visible"), "off")
    close(fig);
end
end
