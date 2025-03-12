package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"paragon" // Replace with actual import path, e.g., "github.com/username/paragon"
)

// Pair defines a word-emoticon pair
type Pair struct {
	Word     string
	Emoticon string
}

// Training data from your provided list
var pairs = []Pair{
	{"acid", "⊂(◉‿◉)つ"},
	{"afraid", "(ㆆ _ ㆆ)"},
	{"alpha", "α"},
	{"angel", "☜(⌒▽⌒)☞"},
	{"angry", "•`_´•"},
	{"arrowhead", "⤜(ⱺ ʖ̯ⱺ)⤏"},
	{"apple", ""},
	{"ass", "(‿|‿)"},
	{"butt", "(‿|‿)"},
	{"awkward", "•͡˘㇁•͡˘"},
	{"bat", "/|\\ ^._.^ /|\\"},
	{"bear", "ʕ·͡ᴥ·ʔ"},
	{"koala", "ʕ·͡ᴥ·ʔ"},
	{"bearflip", "ʕノ•ᴥ•ʔノ ︵ ┻━┻"},
	{"bearhug", "ʕっ•ᴥ•ʔっ"},
	{"because", "∵"},
	{"since", "∵"},
	{"beta", "β"},
	{"bigheart", "❤"},
	{"bitcoin", "₿"},
	{"blackeye", "0__#"},
	{"blubby", "( 0 _ 0 )"},
	{"blush", "(˵ ͡° ͜ʖ ͡°˵)"},
	{"bond", "┌( ͝° ͜ʖ͡°)=ε/̵͇̿̿/’̿’̿ ̿"},
	{"007", "┌( ͝° ͜ʖ͡°)=ε/̵͇̿̿/’̿’̿ ̿"},
	{"boobs", "( . Y . )"},
	{"bored", "(-_-)"},
	{"bribe", "( •͡˘ _•͡˘)ノð"},
	{"bubbles", "( ˘ ³˘)ノ°ﾟº❍｡"},
	{"butterfly", "ƸӜƷ"},
	{"cat", "(= ФェФ=)"},
	{"catlenny", "( ͡° ᴥ ͡°)"},
	{"check", "✔"},
	{"cheer", "※\\(^o^)/※"},
	{"chubby", "╭(ʘ̆~◞౪◟~ʘ̆)╮"},
	{"claro", "(͡ ° ͜ʖ ͡ °)"},
	{"clique", "ヽ༼ ຈل͜ຈ༼ ▀̿̿Ĺ̯̿̿▀̿ ̿༽Ɵ͆ل͜Ɵ͆ ༽ﾉ"},
	{"gang", "ヽ༼ ຈل͜ຈ༼ ▀̿̿Ĺ̯̿̿▀̿ ̿༽Ɵ͆ل͜Ɵ͆ ༽ﾉ"},
	{"squad", "ヽ༼ ຈل͜ຈ༼ ▀̿̿Ĺ̯̿̿▀̿ ̿༽Ɵ͆ل͜Ɵ͆ ༽ﾉ"},
	{"cloud", "☁"},
	{"club", "♣"},
	{"coffee", "c[_]"},
	{"cuppa", "c[_]"},
	{"cmd", "⌘"},
	{"command", "⌘"},
	{"cool", "(•_•) ( •_•)>⌐■-■ (⌐■_■)"},
	{"csi", "(•_•) ( •_•)>⌐■-■ (⌐■_■)"},
	{"copy", "©"},
	{"c", "©"},
	{"creep", "ԅ(≖‿≖ԅ)"},
	{"crim3s", "( ✜︵✜ )"},
	{"cross", "†"},
	{"cry", "(╥﹏╥)"},
	{"crywave", "( ╥﹏╥) ノシ"},
	{"cute", "(｡◕‿‿◕｡)"},
	{"d1", "⚀"},
	{"d2", "⚁"},
	{"d3", "⚂"},
	{"d4", "⚃"},
	{"d5", "⚄"},
	{"d6", "⚅"},
	{"dab", "ヽ( •_)ᕗ"},
	{"damnyou", "(ᕗ ͠° ਊ ͠° )ᕗ"},
	{"dance", "ᕕ(⌐■_■)ᕗ ♪♬"},
	{"dead", "x⸑x"},
	{"dealwithit", "(⌐■_■)"},
	{"dwi", "(⌐■_■)"},
	{"delta", "Δ"},
	{"depressed", "(︶︹︶)"},
	{"derp", "☉ ‿ ⚆"},
	{"diamond", "♦"},
	{"dj", "d[-_-]b"},
	{"dog", "(◕ᴥ◕ʋ)"},
	{"dollar", "$"},
	{"dollarbill", "[̲̅$̲̅(̲̅ιο̲̅̅)̲̅$̲̅]"},
	{"$", "[̲̅$̲̅(̲̅ιο̲̅̅)̲̅$̲̅]"},
	{"dong", "(̿▀̿ ̿Ĺ̯̿̿▀̿ ̿)̄"},
	{"donger", "ヽ༼ຈل͜ຈ༽ﾉ"},
	{"dontcare", "(- ʖ̯-)"},
	{"idc", "(- ʖ̯-)"},
	{"donotwant", "ヽ(｀Д´)ﾉ"},
	{"dontwant", "ヽ(｀Д´)ﾉ"},
	{"dope", "<(^_^)>"},
	{"<<", "«"},
	{">>", "»"},
	{"doubleflat", "𝄫"},
	{"doublesharp", "𝄪"},
	{"doubletableflip", "┻━┻ ︵ヽ(`Д´)ﾉ︵ ┻━┻"},
	{"down", "↓"},
	{"duckface", "(・3・)"},
	{"duel", "ᕕ(╭ರ╭ ͟ʖ╮•́)⊃¤=(————-"},
	{"duh", "(≧︿≦)"},
	{"dunno", "¯\\(°_o)/¯"},
	{"ebola", "ᴇʙᴏʟᴀ"},
	{"eeriemob", "(-(-_-(-_(-_(-_-)_-)-_-)_-)_-)-)"},
	{"ellipsis", "…"},
	{"...", "…"},
	{"emdash", "–"},
	{"--", "–"},
	{"emptystar", "☆"},
	{"emptytriangle", "△"},
	{"t2", "△"},
	{"endure", "(҂◡_◡) ᕤ"},
	{"envelope", "✉︎"},
	{"letter", "✉︎"},
	{"epsilon", "ɛ"},
	{"euro", "€"},
	{"evil", "ψ(｀∇´)ψ"},
	{"evillenny", "(͠≖ ͜ʖ͠≖)"},
	{"excited", "(ﾉ◕ヮ◕)ﾉ*:・ﾟ✧"},
	{"execution", "(⌐■_■)︻╦╤─ (╥﹏╥)"},
	{"facebook", "(╯°□°)╯︵ ʞooqǝɔɐɟ"},
	{"facepalm", "(－‸ლ)"},
	{"fancytext", "вєωαяє, ι αм ƒαη¢у!"},
	{"fart", "(ˆ⺫ˆ๑)<3"},
	{"fight", "(ง •̀_•́)ง"},
	{"finn", "| (• ◡•)|"},
	{"fish", "<\"(((<3"},
	{"5", "卌"},
	{"five", "卌"},
	{"5/8", "⅝"},
	{"flat", "♭"},
	{"bemolle", "♭"},
	{"flexing", "ᕙ(`▽´)ᕗ"},
	{"fliptext", "ǝןqɐʇ ɐ ǝʞıן ǝɯ dıןɟ"},
	{"fliptexttable", "(ノ ゜Д゜)ノ ︵ ǝןqɐʇ ɐ ǝʞıן ʇxǝʇ dıןɟ"},
	{"flower", "(✿◠‿◠)"},
	{"flor", "(✿◠‿◠)"},
	{"f", "✿"},
	{"fly", "─=≡Σ((( つ◕ل͜◕)つ"},
	{"friendflip", "(╯°□°)╯︵ ┻━┻ ︵ ╯(°□° ╯)"},
	{"frown", "(ღ˘⌣˘ღ)"},
	{"fuckoff", "୧༼ಠ益ಠ╭∩╮༽"},
	{"gtfo", "୧༼ಠ益ಠ╭∩╮༽"},
	{"fuckyou", "┌П┐(ಠ_ಠ)"},
	{"fu", "┌П┐(ಠ_ಠ)"},
	{"gentleman", "ಠ_ರೃ"},
	{"sir", "ಠ_ರೃ"},
	{"monocle", "ಠ_ರೃ"},
	{"ghast", "= _ ="},
	{"ghost", "༼ つ ╹ ╹ ༽つ"},
	{"gift", "(´・ω・)っ由"},
	{"present", "(´・ω・)っ由"},
	{"gimme", "༼ つ ◕_◕ ༽つ"},
	{"givemeyourmoney", "(•-•)⌐"},
	{"glitter", "(*・‿・)ノ⌒*:･ﾟ✧"},
	{"glasses", "(⌐ ͡■ ͜ʖ ͡■)"},
	{"glassesoff", "( ͡° ͜ʖ ͡°)ﾉ⌐■-■"},
	{"glitterderp", "(ﾉ☉ヮ⚆)ﾉ ⌒*:･ﾟ✧"},
	{"gloomy", "(_゜_゜_)"},
	{"goatse", "(з๏ε)"},
	{"gotit", "(☞ﾟ∀ﾟ)☞"},
	{"greet", "( ´◔ ω◔`) ノシ"},
	{"greetings", "( ´◔ ω◔`) ノシ"},
	{"gun", "︻╦╤─"},
	{"mg", "︻╦╤─"},
	{"hadouken", "༼つಠ益ಠ༽つ ─=≡ΣO))"},
	{"hammerandsickle", "☭"},
	{"hs", "☭"},
	{"handleft", "☜"},
	{"hl", "☜"},
	{"handright", "☞"},
	{"hr", "☞"},
	{"haha", "٩(^‿^)۶"},
	{"happy", "٩( ๑╹ ꇴ╹)۶"},
	{"happygarry", "ᕕ( ᐛ )ᕗ"},
	{"h", "♥"},
	{"heart", "♥"},
	{"hello", "(ʘ‿ʘ)╯"},
	{"ohai", "(ʘ‿ʘ)╯"},
	{"bye", "(ʘ‿ʘ)╯"},
	{"help", "\\(°Ω°)/"},
	{"highfive", "._.)/\\(._."},
	{"hitting", "( ｀皿´)｡ﾐ/"},
	{"hug", "(づ｡◕‿‿◕｡)づ"},
	{"hugs", "(づ｡◕‿‿◕｡)づ"},
	{"iknowright", "┐｜･ิω･ิ#｜┌"},
	{"ikr", "┐｜･ิω･ิ#｜┌"},
	{"illuminati", "୧(▲ᴗ▲)ノ"},
	{"infinity", "∞"},
	{"inf", "∞"},
	{"inlove", "(っ´ω`c)♡"},
	{"int", "∫"},
	{"internet", "ଘ(੭*ˊᵕˋ)੭* ̀ˋ ɪɴᴛᴇʀɴᴇᴛ"},
	{"interrobang", "‽"},
	{"jake", "(❍ᴥ❍ʋ)"},
	{"kappa", "(¬,‿,¬)"},
	{"kawaii", "≧◡≦"},
	{"keen", "┬┴┬┴┤Ɵ͆ل͜Ɵ͆ ༽ﾉ"},
	{"kiahh", "~\\(≧▽≦)/~"},
	{"kiss", "(づ ￣ ³￣)づ"},
	{"kyubey", "／人◕ ‿‿ ◕人＼"},
	{"lambda", "λ"},
	{"lazy", "_(:3」∠)_"},
	{"left", "←"},
	{"<-", "←"},
	{"lenny", "( ͡° ͜ʖ ͡°)"},
	{"lennybill", "[̲̅$̲̅(̲̅ ͡° ͜ʖ ͡°̲̅)̲̅$̲̅]"},
	{"lennyfight", "(ง ͠° ͟ʖ ͡°)ง"},
	{"lennyflip", "(ノ ͡° ͜ʖ ͡°ノ) ︵ ( ͜。 ͡ʖ ͜。)"},
	{"lennygang", "( ͡°( ͡° ͜ʖ( ͡° ͜ʖ ͡°)ʖ ͡°) ͡°)"},
	{"lennyshrug", "¯\\_( ͡° ͜ʖ ͡°)_/¯"},
	{"lennysir", "( ಠ ͜ʖ ರೃ)"},
	{"lennystalker", "┬┴┬┴┤( ͡° ͜ʖ├┬┴┬┴"},
	{"lennystrong", "ᕦ( ͡° ͜ʖ ͡°)ᕤ"},
	{"lennywizard", "╰( ͡° ͜ʖ ͡° )つ──☆*:・ﾟ"},
	{"loading", "███▒▒▒▒▒▒▒"},
	{"lol", "L(° O °L)"},
	{"look", "(ಡ_ಡ)☞"},
	{"loud", "ᕦ(⩾﹏⩽)ᕥ"},
	{"noise", "ᕦ(⩾﹏⩽)ᕥ"},
	{"love", "♥‿♥"},
	{"lovebear", "ʕ♥ᴥ♥ʔ"},
	{"lumpy", "꒰ ꒡⌓꒡꒱"},
	{"luv", "-`ღ´-"},
	{"magic", "ヽ(｀Д´)⊃━☆ﾟ. * ･ ｡ﾟ,"},
	{"magicflip", "(/¯◡ ‿ ◡)/¯ ~ ┻━┻"},
	{"meep", "\\(°^°)/"},
	{"meh", "ಠ_ಠ"},
	{"metal", "\\m/,(> . <)_\\m/"},
	{"rock", "\\m/,(> . <)_\\m/"},
	{"mistyeyes", "ಡ_ಡ"},
	{"monster", "༼ ༎ຶ ෴ ༎ຶ༽"},
	{"natural", "♮"},
	{"needle", "┌(◉ ͜ʖ◉)つ┣▇▇▇═──"},
	{"inject", "┌(◉ ͜ʖ◉)つ┣▇▇▇═──"},
	{"nerd", "(⌐⊙_⊙)"},
	{"nice", "( ͡° ͜ °)"},
	{"no", "→_←"},
	{"noclue", "／人◕ __ ◕人＼"},
	{"nom", "(っˆڡˆς)"},
	{"yummy", "(っˆڡˆς)"},
	{"delicious", "(っˆڡˆς)"},
	{"note", "♫"},
	{"sing", "♫"},
	{"nuclear", "☢"},
	{"radioactive", "☢"},
	{"nukular", "☢"},
	{"nyan", "~=[,,_,,]:3"},
	{"nyeh", "@^@"},
	{"ohshit", "( º﹃º )"},
	{"omega", "Ω"},
	{"omg", "◕_◕"},
	{"1/8", "⅛"},
	{"1/4", "¼"},
	{"1/2", "½"},
	{"1/3", "⅓"},
	{"opt", "⌥"},
	{"option", "⌥"},
	{"orly", "(눈_눈)"},
	{"ohyou", "(◞థ౪థ)ᴖ"},
	{"ou", "(◞థ౪థ)ᴖ"},
	{"peace", "✌(-‿-)✌"},
	{"victory", "✌(-‿-)✌"},
	{"pear", "(__>-"},
	{"pi", "π"},
	{"pingpong", "( •_•)O*¯`·.¸.·´¯`°Q(•_• )"},
	{"plain", "._."},
	{"pleased", "(˶‾᷄ ⁻̫ ‾᷅˵)"},
	{"point", "(☞ﾟヮﾟ)☞"},
	{"pooh", "ʕ •́؈•̀)"},
	{"porcupine", "(•ᴥ• )́`́'́`́'́⻍"},
	{"pound", "£"},
	{"praise", "(☝ ՞ਊ ՞)☝"},
	{"punch", "O=('-'Q)"},
	{"rage", "t(ಠ益ಠt)"},
	{"mad", "t(ಠ益ಠt)"},
	{"rageflip", "(ノಠ益ಠ)ノ彡┻━┻"},
	{"rainbowcat", "(=^･ｪ･^=))ﾉ彡☆"},
	{"really", "ò_ô"},
	{"r", "®"},
	{"right", "→"},
	{"->", "→"},
	{"riot", "୧༼ಠ益ಠ༽୨"},
	{"rolldice", "⚃"},
	{"rolleyes", "(◔_◔)"},
	{"rose", "✿ڿڰۣ—"},
	{"run", "(╯°□°)╯"},
	{"sad", "ε(´סּ︵סּ`)з"},
	{"saddonger", "ヽ༼ຈʖ̯ຈ༽ﾉ"},
	{"sadlenny", "( ͡° ʖ̯ ͡°)"},
	{"7/8", "⅞"},
	{"sharp", "♯"},
	{"diesis", "♯"},
	{"shout", "╚(•⌂•)╝"},
	{"shrug", "¯\\_(ツ)_/¯"},
	{"shy", "=^_^="},
	{"sigma", "Σ"},
	{"sum", "Σ"},
	{"skull", "☠"},
	{"smile", "ツ"},
	{"smiley", "☺︎"},
	{"smirk", "¬‿¬"},
	{"snowman", "☃"},
	{"sob", "(;´༎ຶД༎ຶ`)"},
	{"soviettableflip", "ノ┬─┬ノ ︵ ( \\o°o)\\"},
	{"spade", "♠"},
	{"sqrt", "√"},
	{"squid", "<コ:彡"},
	{"star", "★"},
	{"strong", "ᕙ(⇀‸↼‶)ᕗ"},
	{"suicide", "ε/̵͇̿̿/’̿’̿ ̿(◡︵◡)"},
	{"sum", "∑"},
	{"sun", "☀"},
	{"surprised", "(๑•́ ヮ •̀๑)"},
	{"surrender", "\\_(-_-)_/"},
	{"stalker", "┬┴┬┴┤(･_├┬┴┬┴"},
	{"swag", "(̿▀̿‿ ̿▀̿ ̿)"},
	{"sword", "o()xxxx[{::::::::::::::::::>"},
	{"tableflip", "(ノ ゜Д゜)ノ ︵ ┻━┻"},
	{"tau", "τ"},
	{"tears", "(ಥ﹏ಥ)"},
	{"terrorist", "୧༼ಠ益ಠ༽︻╦╤─"},
	{"thanks", "\\(^-^)/"},
	{"thankyou", "\\(^-^)/"},
	{"ty", "\\(^-^)/"},
	{"therefore", "⸫"},
	{"so", "⸫"},
	{"this", "( ͡° ͜ʖ ͡°)_/¯"},
	{"3/8", "⅜"},
	{"tiefighter", "|=-(¤)-=|"},
	{"tired", "(=____=)"},
	{"toldyouso", "☜(꒡⌓꒡)"},
	{"toldyou", "☜(꒡⌓꒡)"},
	{"toogood", "ᕦ(òᴥó)ᕥ"},
	{"tm", "™"},
	{"triangle", "▲"},
	{"t", "▲"},
	{"2/3", "⅔"},
	{"unflip", "┬──┬ ノ(ò_óノ)"},
	{"up", "↑"},
	{"victory", "(๑•̀ㅂ•́)ง✧"},
	{"wat", "(ÒДÓױ)"},
	{"wave", "( * ^ *) ノシ"},
	{"whaa", "Ö"},
	{"whistle", "(っ^з^)♪♬"},
	{"whoa", "(°o•)"},
	{"why", "ლ(`◉◞౪◟◉‵ლ)"},
	{"witchtext", "WHΣИ $HΛLL WΣ †HЯΣΣ MΣΣ† ΛGΛ|И?"},
	{"woo", "＼(＾O＾)／"},
	{"wtf", "(⊙＿⊙')"},
	{"wut", "⊙ω⊙"},
	{"yay", "\\( ﾟヮﾟ)/"},
	{"yeah", "(•̀ᴗ•́)و ̑̑"},
	{"yes", "(•̀ᴗ•́)و ̑̑"},
	{"yen", "¥"},
	{"yinyang", "☯"},
	{"yy", "☯"},
	{"yolo", "Yᵒᵘ Oᶰˡʸ Lᶤᵛᵉ Oᶰᶜᵉ"},
	{"youkids", "ლ༼>╭ ͟ʖ╮<༽ლ"},
	{"ukids", "ლ༼>╭ ͟ʖ╮<༽ლ"},
	{"yuno", "(屮ﾟДﾟ)屮 Y U NO"},
	{"zen", "⊹╰(⌣ʟ⌣)╯⊹"},
	{"meditation", "⊹╰(⌣ʟ⌣)╯⊹"},
	{"omm", "⊹╰(⌣ʟ⌣)╯⊹"},
	{"zoidberg", "(V) (°,,,,°) (V)"},
	{"zombie", "[¬º-°]¬"},
}

// -------------------------------------------------------
// 1) Local helper "Char-level" Encode/Decode
// -------------------------------------------------------
func encodeCharLevel(tok *paragon.CustomTokenizer, text string) []int {
	ids := make([]int, 0, len(text))
	padID := tok.Vocab["[PAD]"]
	for _, char := range text {
		c := string(char)
		if id, ok := tok.Vocab[c]; ok {
			ids = append(ids, id)
		} else {
			ids = append(ids, padID)
		}
	}
	return ids
}

func decodeCharLevel(tok *paragon.CustomTokenizer, ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if str, ok := tok.ReverseVocab[id]; ok {
			// Skip if it's a special token
			if !tok.SpecialTokens[id] {
				sb.WriteString(str)
			}
		}
	}
	return sb.String()
}

// -------------------------------------------------------
// 2) Partial masking AFTER [SEP] only
// -------------------------------------------------------
func betterAddNoiseWithSep(d *paragon.DiffusionModel, x0 []int, t int, sepPos int) []int {
	noisy := make([]int, len(x0))
	copy(noisy, x0)

	padID := d.Tokenizer.Vocab["[PAD]"]
	maskID := d.Tokenizer.Vocab["[MASK]"]
	fraction := d.MaskFraction[t]
	if fraction <= 0 {
		return noisy
	}

	// Only consider tokens after [SEP], ignoring pads
	var idxes []int
	for i := sepPos + 1; i < len(x0); i++ {
		if x0[i] != padID {
			idxes = append(idxes, i)
		}
	}
	rand.Shuffle(len(idxes), func(i, j int) { idxes[i], idxes[j] = idxes[j], idxes[i] })

	k := int(math.Round(float64(len(idxes)) * fraction))
	for i := 0; i < k && i < len(idxes); i++ {
		noisy[idxes[i]] = maskID
	}
	return noisy
}

// -------------------------------------------------------
//  3. Batched training: partial masking after [SEP], single backward per batch,
//     measure accuracy, generate examples each epoch. Uses BackwardExternal.
//
// -------------------------------------------------------
func trainBetterDiffusionWithSepBatch(d *paragon.DiffusionModel, samples [][]int, sepPositions []int) {
	data := make([][]int, len(samples))
	copy(data, samples)

	batchSize := 4 // Adjust as desired
	//baseLR := d.Config.LearningRate

	for epoch := 0; epoch < d.Config.Epochs; epoch++ {
		// Simple linear LR decay
		progress := float64(epoch) / float64(d.Config.Epochs)
		lr := d.Config.LearningRate * (math.Cos(progress*math.Pi) + 1) / 2

		// Shuffle data
		rand.Shuffle(len(data), func(i, j int) {
			data[i], data[j] = data[j], data[i]
			sepPositions[i], sepPositions[j] = sepPositions[j], sepPositions[i]
		})

		totalLoss := 0.0
		maskedCorrect := 0
		maskedCount := 0

		// Batch loop
		for i := 0; i < len(data); i += batchSize {
			end := i + batchSize
			if end > len(data) {
				end = len(data)
			}
			batch := data[i:end]
			batchSep := sepPositions[i:end]

			// We'll accumulate error terms for the entire batch, then call BackwardExternal once
			accumError := make([]float64, d.Config.MaxLength*d.Tokenizer.VocabSize)

			localLoss := 0.0
			for idx, x0 := range batch {
				sepPos := batchSep[idx]
				t := rand.Intn(d.Config.NumTimesteps)
				xt := betterAddNoiseWithSep(d, x0, t, sepPos)

				// Build one-hot
				batchInput := make([][]float64, d.Config.MaxLength)
				for k, tok := range xt {
					row := make([]float64, d.Tokenizer.VocabSize)
					if tok >= 0 && tok < d.Tokenizer.VocabSize {
						row[tok] = 1.0
					}
					batchInput[k] = row
				}

				output2D := d.Network.ForwardTransformer(batchInput)
				preds := output2D[0] // shape = [MaxLength * VocabSize]

				maskID := d.Tokenizer.Vocab["[MASK]"]
				for pos, tok := range xt {
					// Only compute loss & accuracy for positions after [SEP] that are masked
					if pos <= sepPos {
						continue
					}
					if tok == maskID {
						start := pos * d.Tokenizer.VocabSize
						endPos := start + d.Tokenizer.VocabSize
						probs := paragon.Softmax(preds[start:endPos])
						target := x0[pos]

						// Cross-entropy
						localLoss -= math.Log(math.Max(probs[target], 1e-10))

						// For accuracy, pick argmax
						best := 0
						bestProb := probs[0]
						for m := 1; m < len(probs); m++ {
							if probs[m] > bestProb {
								bestProb = probs[m]
								best = m
							}
						}
						if best == target {
							maskedCorrect++
						}
						maskedCount++

						// Accumulate error terms
						for m := 0; m < d.Tokenizer.VocabSize; m++ {
							delta := probs[m]
							if m == target {
								delta -= 1
							}
							// gradient clip
							if delta > 5.0 {
								delta = 5.0
							} else if delta < -5.0 {
								delta = -5.0
							}
							accumError[start+m] += delta
						}
					}
				}
			}

			// Average loss over the batch
			localLoss /= float64(len(batch))
			totalLoss += localLoss

			// Now do one backward pass for this batch
			shaped := make([][]float64, d.Config.MaxLength)
			for k := 0; k < d.Config.MaxLength; k++ {
				start := k * d.Tokenizer.VocabSize
				shaped[k] = accumError[start : start+d.Tokenizer.VocabSize]
			}
			d.Network.BackwardExternal(shaped, lr)
		}

		epochLoss := totalLoss / float64(len(data)/batchSize)
		accuracy := 0.0
		if maskedCount > 0 {
			accuracy = float64(maskedCorrect) / float64(maskedCount)
		}

		// Print progress
		fmt.Printf("Epoch %d | LR: %.5f | Loss: %.4f | Masked Acc: %.2f%%\n",
			epoch, lr, epochLoss, accuracy*100.0)

		// Generate sample emoticons each epoch
		if epoch%1 == 0 { // or pick a different interval
			words := []string{"happy", "sad"}
			fmt.Println("Sample generations:")
			for _, w := range words {
				g := generateEmoticon(d, w)
				fmt.Printf("   %s => %s\n", w, g)
			}
			fmt.Println()
		}

		if err := d.Network.SaveToGob("emoticon_model.gob"); err != nil {
			panic(fmt.Errorf("failed to save model to gob: %v", err))
		}

		// Early stop if accuracy >= 95%
		if accuracy >= 0.95 {
			fmt.Println("Early stopping: Reached 95% masked accuracy!")
			break
		}
	}
}

// -------------------------------------------------------
// 4) Single-pass reverse diffusion generator
// -------------------------------------------------------
func generateEmoticon(d *paragon.DiffusionModel, inputWord string) string {
	sepID := d.Tokenizer.Vocab["[SEP]"]
	maskID := d.Tokenizer.Vocab["[MASK]"]
	padID := d.Tokenizer.Vocab["[PAD]"]

	wordIDs := encodeCharLevel(d.Tokenizer, inputWord)
	seq := append(wordIDs, sepID)
	sepPos := len(wordIDs)

	// Fill up to MaxLength with [MASK]
	for len(seq) < d.Config.MaxLength {
		seq = append(seq, maskID)
	}

	// Reverse diffusion
	for t := d.Config.NumTimesteps - 1; t >= 0; t-- {
		// build one-hot
		batchInput := make([][]float64, d.Config.MaxLength)
		for i, tok := range seq {
			row := make([]float64, d.Tokenizer.VocabSize)
			if tok >= 0 && tok < d.Tokenizer.VocabSize {
				row[tok] = 1.0
			}
			batchInput[i] = row
		}
		output2D := d.Network.ForwardTransformer(batchInput)
		preds := output2D[0]

		// fill in any [MASK] after [SEP]
		for i := sepPos + 1; i < d.Config.MaxLength; i++ {
			if seq[i] == maskID {
				start := i * d.Tokenizer.VocabSize
				end := start + d.Tokenizer.VocabSize
				probs := paragon.Softmax(preds[start:end])
				// Just pick argmax or threshold
				best := 0
				bestProb := probs[0]
				for m := 1; m < len(probs); m++ {
					if probs[m] > bestProb {
						bestProb = probs[m]
						best = m
					}
				}
				seq[i] = best
			}
		}
		// Optional re-masking for multi-step generation:
		// if t > 0 { ... re-mask with some probability ... }
	}

	// Gather emoticon portion
	emoticonIDs := []int{}
	for i := sepPos + 1; i < d.Config.MaxLength; i++ {
		if seq[i] == padID {
			break
		}
		emoticonIDs = append(emoticonIDs, seq[i])
	}
	return decodeCharLevel(d.Tokenizer, emoticonIDs)
}

// -------------------------------------------------------
// main()
// -------------------------------------------------------
func main() {
	rand.Seed(time.Now().UnixNano())

	// 1) Build a custom char-level tokenizer with [PAD], [MASK], [SEP]
	tok := &paragon.CustomTokenizer{
		Vocab:         make(map[string]int),
		ReverseVocab:  make(map[int]string),
		SpecialTokens: make(map[int]bool),
	}
	specials := []string{"[PAD]", "[MASK]", "[SEP]"}
	for i, s := range specials {
		tok.Vocab[s] = i
		tok.ReverseVocab[i] = s
		tok.SpecialTokens[i] = true
	}
	nextID := len(specials)
	// Add ASCII
	for c := rune(' '); c <= '~'; c++ {
		ch := string(c)
		if _, exists := tok.Vocab[ch]; !exists {
			tok.Vocab[ch] = nextID
			tok.ReverseVocab[nextID] = ch
			nextID++
		}
	}
	// Add unique chars from your pairs
	for _, p := range pairs {
		for _, r := range p.Word + p.Emoticon {
			c := string(r)
			if _, ok := tok.Vocab[c]; !ok {
				tok.Vocab[c] = nextID
				tok.ReverseVocab[nextID] = c
				nextID++
			}
		}
	}
	tok.VocabSize = nextID

	// 2) Compute maxSeqLen
	maxSeqLen := 0
	for _, p := range pairs {
		wLen := len([]rune(p.Word))
		eLen := len([]rune(p.Emoticon))
		seqLen := wLen + 1 + eLen
		if seqLen > maxSeqLen {
			maxSeqLen = seqLen
		}
	}

	// 3) Create config
	tConfig := paragon.TransformerConfig{
		DModel:      32,
		NHeads:      2,
		NLayers:     2,
		FeedForward: 32,
		VocabSize:   tok.VocabSize,
		MaxLength:   maxSeqLen,
		Activation:  "relu",
	}
	dConfig := paragon.DiffusionConfig{
		NumTimesteps:      5,
		MaxLength:         maxSeqLen,
		LearningRate:      0.01,
		Epochs:            1000, // reduce epochs for demo
		Temperature:       0.8,
		TopK:              1,
		MaskScheduleStart: 0.1,
		MaskScheduleEnd:   0.9,
	}

	// 4) Build network + model
	network := paragon.NewTransformerEncoder(tConfig)
	model := paragon.NewDiffusionModel(network, dConfig, []string{})
	model.Tokenizer = tok

	// 5) Prepare training samples
	sepID := tok.Vocab["[SEP]"]
	padID := tok.Vocab["[PAD]"]
	data := make([][]int, len(pairs))
	sepPositions := make([]int, len(pairs))
	for i, p := range pairs {
		wIDs := encodeCharLevel(tok, p.Word)
		eIDs := encodeCharLevel(tok, p.Emoticon)
		seq := append(wIDs, sepID)
		sepPos := len(wIDs)
		seq = append(seq, eIDs...)
		if len(seq) < maxSeqLen {
			padNeeded := maxSeqLen - len(seq)
			padding := make([]int, padNeeded)
			for j := range padding {
				padding[j] = padID
			}
			seq = append(seq, padding...)
		}
		data[i] = seq
		sepPositions[i] = sepPos
	}

	// 6) Train or load
	modelFile := "emoticon_model.gob"
	if _, err := os.Stat(modelFile); os.IsNotExist(err) {
		fmt.Println("No model file found; training with partial-masking after [SEP] ...")
		trainBetterDiffusionWithSepBatch(model, data, sepPositions)
		fmt.Println("Training complete. (Not saving for this demo.)")
	} else {
		fmt.Println("Model file found. For demo, we will just re-train anyway.")
		if err := model.Network.LoadFromGob(modelFile); err != nil {
			panic(fmt.Errorf("failed to load model from gob: %v", err))
		}
		trainBetterDiffusionWithSepBatch(model, data, sepPositions)
	}

	// 7) Generate final test
	fmt.Println("\nFinal test emoticons:")
	testWords := []string{"happy", "sad", "excited", "unknown"}
	for _, w := range testWords {
		gen := generateEmoticon(model, w)
		fmt.Printf("  Input: %s => %s\n", w, gen)
	}
}
