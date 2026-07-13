-- callouts.lua — map labeled blockquotes to colored tcolorbox callouts.
-- A blockquote whose first paragraph starts with a bold label (직관/기호/시스템 모델링/예시/주의/핵심)
-- becomes a colored box; the label is used as the box title.
local ENV = {
  ["직관"] = "cboxintuition", ["물리적 직관"] = "cboxintuition",
  ["기호"] = "cboxsymbol", ["기호 풀이"] = "cboxsymbol", ["기호풀이"] = "cboxsymbol",
  ["시스템 모델링"] = "cboxsysmodel", ["시스템 모델링 관점"] = "cboxsysmodel", ["시스템"] = "cboxsysmodel",
  ["예시"] = "cboxexample", ["예제"] = "cboxexample", ["비유"] = "cboxexample",
  ["주의"] = "cboxnote", ["한계"] = "cboxnote", ["함정"] = "cboxnote",
  ["핵심"] = "cboxkey", ["요약"] = "cboxkey", ["한 줄 요약"] = "cboxkey",
}

function BlockQuote(el)
  local first = el.content[1]
  if not (first and first.t == "Para" and first.content[1] and first.content[1].t == "Strong") then
    return nil
  end
  local raw = pandoc.utils.stringify(first.content[1])
  local label = raw:gsub("%s*[.:]%s*$", ""):gsub("^%s+", ""):gsub("%s+$", "")
  local env = ENV[label]
  if not env then return nil end
  -- drop the leading Strong label (and a following space/period) from the first paragraph
  table.remove(first.content, 1)
  while first.content[1] and (first.content[1].t == "Space" or
        (first.content[1].t == "Str" and (first.content[1].text == "." or first.content[1].text == ":"))) do
    table.remove(first.content, 1)
  end
  local body = pandoc.write(pandoc.Pandoc(el.content), "latex")
  return pandoc.RawBlock("latex", "\\begin{" .. env .. "}{" .. label .. "}\n" .. body .. "\n\\end{" .. env .. "}")
end
