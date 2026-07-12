-- breakable.lua — insert line-break opportunities into long unbreakable Latin/code runs.
--
-- Why this exists (method note): in a luatexja CJK document, long Latin token runs joined by
-- '/', '·', ':' (name lists like GLA/DeltaNet/GDN, arXiv IDs like arXiv:2501.00663, code
-- identifiers like update_in_place/mark_dirty) are NOT reliably reported as "Overfull \hbox";
-- luatexja lets them run past the right margin silently. So the build log is not a valid overflow
-- gate — a full-page pixel scan (build/overflow_gate.py) is. This filter is the systematic fix:
-- it inserts \allowbreak after the joiners so the line breaker can wrap them.

local BREAK_STR  = { ['/']=true, ['·']=true, [':']=true }          -- break after these in prose tokens
local BREAK_CODE = { ['/']=true, ['·']=true, ['_']=true, [':']=true, ['.']=true }  -- and in code

local function is_candidate(t)
  return t:find('[/·]') or t:find('arXiv') or t:find('%d%.%d%d') ~= nil
end

-- prose Str: split into characters, re-emit as Str pieces with \allowbreak after joiners.
-- Splitting per char keeps pandoc's own escaping of each piece intact.
function Str(el)
  local t = el.text
  if not is_candidate(t) then return nil end
  local out = {}
  local arxiv = t:find('arXiv') ~= nil or t:find('%d%.%d%d') ~= nil
  for _, code in utf8.codes(t) do
    local ch = utf8.char(code)
    table.insert(out, pandoc.Str(ch))
    if BREAK_STR[ch] or (arxiv and ch == '.') then
      table.insert(out, pandoc.RawInline('latex', '\\allowbreak{}'))
    end
  end
  return out
end

-- inline Code: emit \texttt{...} with LaTeX-escaped content and \allowbreak after code joiners.
local function esc(ch)
  local map = {
    ['\\']='\\textbackslash{}', ['{']='\\{', ['}']='\\}', ['$']='\\$', ['&']='\\&',
    ['#']='\\#', ['%']='\\%', ['_']='\\_', ['~']='\\textasciitilde{}', ['^']='\\textasciicircum{}',
  }
  return map[ch] or ch
end

function Code(el)
  local buf = { '\\texttt{' }
  for _, code in utf8.codes(el.text) do
    local ch = utf8.char(code)
    buf[#buf+1] = esc(ch)
    if BREAK_CODE[ch] then buf[#buf+1] = '\\allowbreak{}' end
  end
  buf[#buf+1] = '}'
  return pandoc.RawInline('latex', table.concat(buf))
end
