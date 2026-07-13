-- mathfit.lua — shrink any display equation that would exceed \linewidth (preserve aspect/no clip).
-- Wraps DisplayMath in \adjustbox{max width=\linewidth}: narrow equations are unchanged, wide ones
-- scale down to fit. General fix for long chained equations in the translations/booklets.
function Math(el)
  if el.mathtype == "DisplayMath" then
    return pandoc.RawInline("latex",
      "\\[\\adjustbox{max width=\\linewidth}{$\\displaystyle " .. el.text .. "$}\\]")
  end
  return nil
end
