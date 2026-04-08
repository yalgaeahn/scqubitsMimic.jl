module ScQubitsMimicExampleMakie

using CairoMakie

const Makie = CairoMakie.Makie
const FTA = Makie.FreeTypeAbstraction
const HANGUL_SAMPLES = ('가', '한', '글')
const DEFAULT_FONT_CANDIDATES = [
    "Apple SD Gothic Neo",
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "NanumGothic",
    "Nanum Gothic",
    "Noto Sans CJK KR",
    "Noto Sans KR",
]

function _font_candidates()
    candidates = String[]
    env_font = strip(get(ENV, "SCQUBITSMIMIC_MAKIE_FONT", ""))
    !isempty(env_font) && push!(candidates, env_font)
    append!(candidates, DEFAULT_FONT_CANDIDATES)
    return unique(candidates)
end

function _supports_hangul(font)
    return all(ch -> FTA.glyph_index(font, ch) != 0, HANGUL_SAMPLES)
end

function _resolve_hangul_font()
    for candidate in _font_candidates()
        font = try
            Makie.to_font(candidate)
        catch
            nothing
        end
        font === nothing && continue
        _supports_hangul(font) && return (candidate, font)
    end

    error(
        "Could not find a Hangul-capable font for CairoMakie. " *
        "Set SCQUBITSMIMIC_MAKIE_FONT to a system font name or font file path."
    )
end

function setup_makie_font!()
    candidate, font = _resolve_hangul_font()

    Makie.update_theme!(
        font = :regular,
        fonts = Makie.Attributes(
            regular = font,
            bold = font,
            italic = font,
            bolditalic = font,
            bold_italic = font,
        ),
        Axis = (
            titlefont = :bold,
            subtitlefont = :regular,
            xlabelfont = :regular,
            ylabelfont = :regular,
            xticklabelfont = :regular,
            yticklabelfont = :regular,
        ),
        Axis3 = (
            titlefont = :bold,
            xlabelfont = :regular,
            ylabelfont = :regular,
            zlabelfont = :regular,
            xticklabelfont = :regular,
            yticklabelfont = :regular,
            zticklabelfont = :regular,
        ),
        Legend = (
            titlefont = :bold,
            labelfont = :regular,
        ),
        Colorbar = (
            labelfont = :regular,
            ticklabelfont = :regular,
        ),
    )

    return candidate
end

end # module
