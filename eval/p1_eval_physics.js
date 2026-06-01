// ================================
// POST-REQUEST / TESTS SCRIPT
// ================================
// Mỗi iteration chỉ có 1 test duy nhất:
// PASS khi:
// 1. Status code = 200
// 2. Response parse được JSON
// 3. Response có field answer
// 4. Gold/API answer comparable
// 5. Answer match
//
// Supports:
// - Yes/No, True/False
// - Numeric answer + unit normalization
// - Dataset format: answer + unit
// - Math expressions: 9\sqrt{3} × 10^-27
// - SI prefixes m/u/n/p cho tất cả unit prefixable đang support
// - String answer: API answer nằm trong gold answer thì tính đúng
// - ±10% tolerance after normalization

const REL_TOLERANCE = 0.10;
const ABS_TOLERANCE_FLOOR = 1e-12;
const MAX_WRONG_CASES = 100;

function decodeEscapedUnicode(value) {
    if (value === undefined || value === null) return "";
    return String(value).replace(/\\u([0-9a-fA-F]{4})/g, function (_, hex) {
        return String.fromCharCode(parseInt(hex, 16));
    });
}

function valueToText(value) {
    if (value === undefined || value === null) return "";
    if (Array.isArray(value)) return value.map(valueToText).join(" ");
    if (typeof value === "object") return decodeEscapedUnicode(JSON.stringify(value));
    return decodeEscapedUnicode(value);
}

function getCollectionNumber(name) {
    return Number(pm.collectionVariables.get(name) || 0);
}

function setCollectionNumber(name, value) {
    pm.collectionVariables.set(name, String(value));
}

function combineAnswerAndUnit(answer, unit) {
    const answerText = valueToText(answer).trim();
    const unitText = valueToText(unit).trim();

    if (!answerText) return answerText;
    if (!unitText || unitText === "-") return answerText;

    const escapedUnit = unitText.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    const hasUnitAlready = new RegExp(`(^|\\s)${escapedUnit}(\\s|$)`, "i").test(answerText);

    if (hasUnitAlready) return answerText;

    return `${answerText} ${unitText}`;
}

function normalizeForBoolean(text) {
    return valueToText(text)
        .toLowerCase()
        .replace(/[`*_~]/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function extractBoolean(value) {
    const text = normalizeForBoolean(value);
    if (!text) return null;

    const explicit = text.match(
        /(?:answer|result|final answer|the answer is|therefore|thus)\s*(?:is|:)?\s*\b(yes|no|true|false)\b/i
    );

    if (explicit) {
        const token = explicit[1].toLowerCase();
        return token === "yes" || token === "true";
    }

    if (/^\s*(yes|true)\b/.test(text)) return true;
    if (/^\s*(no|false)\b/.test(text)) return false;

    const compact = text.replace(/[^a-zà-ỹđ]+/g, " ").trim();

    if (/^(yes|true)$/.test(compact)) return true;
    if (/^(no|false)$/.test(compact)) return false;

    if (/^(có|co|đúng|dung)$/.test(compact)) return true;
    if (/^(không|khong|sai)$/.test(compact)) return false;

    return null;
}

function normalizeSuperscript(text) {
    const map = {
        "⁰": "0",
        "¹": "1",
        "²": "2",
        "³": "3",
        "⁴": "4",
        "⁵": "5",
        "⁶": "6",
        "⁷": "7",
        "⁸": "8",
        "⁹": "9",
        "⁻": "-",
        "⁺": "+"
    };

    return String(text).replace(/[⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺]/g, ch => map[ch] || ch);
}

function normalizeMathText(value) {
    return normalizeSuperscript(valueToText(value))
        .replace(/,/g, "")
        .replace(/\\\\/g, "\\")
        .replace(/\\times/g, "*")
        .replace(/×/g, "*")
        .replace(/∙|·/g, "*")
        .replace(/−/g, "-")
        .replace(/\\cdot/g, "*")
        .replace(/\\\(|\\\)/g, " ")
        .replace(/\$/g, " ")
        .replace(/\s+/g, " ")
        .trim();
}

const PREFIX_WORD_TO_SYMBOL = {
    pico: "p",
    nano: "n",
    micro: "u",
    milli: "m",
    centi: "c",
    kilo: "k",
    mega: "M",
    giga: "G"
};

const WORD_UNIT_PATTERNS = [
    { pattern: "meters?|metres?", symbol: "m" },
    { pattern: "grams?", symbol: "g" },
    { pattern: "seconds?|secs?", symbol: "s" },
    { pattern: "hertz", symbol: "Hz" },
    { pattern: "volts?", symbol: "V" },
    { pattern: "amperes?|amps?", symbol: "A" },
    { pattern: "watts?", symbol: "W" },
    { pattern: "joules?", symbol: "J" },
    { pattern: "newtons?", symbol: "N" },
    { pattern: "coulombs?", symbol: "C" },
    { pattern: "farads?", symbol: "F" },
    { pattern: "henrys?", symbol: "H" },
    { pattern: "ohms?", symbol: "ohm" },
    { pattern: "pascals?", symbol: "Pa" },
    { pattern: "teslas?", symbol: "T" }
];

function replacePrefixedUnitWords(text) {
    let output = text;
    const prefixPattern = Object.keys(PREFIX_WORD_TO_SYMBOL).join("|");

    for (const unit of WORD_UNIT_PATTERNS) {
        const regex = new RegExp(`\\b(${prefixPattern})\\s*-?\\s*(${unit.pattern})\\b`, "gi");
        output = output.replace(regex, function (_, prefix) {
            return PREFIX_WORD_TO_SYMBOL[String(prefix).toLowerCase()] + unit.symbol;
        });
    }

    return output;
}

function replacePlainUnitWords(text) {
    return text
        .replace(/kilograms?/gi, "kg")
        .replace(/grams?/gi, "g")
        .replace(/meters?|metres?/gi, "m")
        .replace(/seconds?|secs?/gi, "s")
        .replace(/minutes?|mins?/gi, "min")
        .replace(/hours?|hrs?/gi, "hr")
        .replace(/hertz/gi, "Hz")
        .replace(/volts?/gi, "V")
        .replace(/amperes?|amps?/gi, "A")
        .replace(/watts?/gi, "W")
        .replace(/joules?/gi, "J")
        .replace(/newtons?/gi, "N")
        .replace(/coulombs?/gi, "C")
        .replace(/farads?/gi, "F")
        .replace(/henrys?/gi, "H")
        .replace(/ohms?/gi, "ohm")
        .replace(/pascals?/gi, "Pa")
        .replace(/teslas?/gi, "T")
        .replace(/degrees?/gi, "deg")
        .replace(/radians?/gi, "rad");
}

function normalizeUnitText(rawUnit) {
    if (!rawUnit) return "";

    let u = valueToText(rawUnit)
        .trim()
        .replace(/[.,;:!?]+$/g, "")
        .replace(/\\Omega/g, "ohm")
        .replace(/Ω|Ω/g, "ohm")
        .replace(/ω/g, "ohm")
        .replace(/\\mu/g, "u")
        .replace(/μ|µ/g, "u")
        .replace(/²/g, "^2")
        .replace(/³/g, "^3")
        .replace(/\bper\b/gi, "/")
        .replace(/[()\[\]{}]/g, "");

    u = replacePrefixedUnitWords(u);
    u = replacePlainUnitWords(u);

    return u
        .replace(/\s*(\/|\*|·)\s*/g, "$1")
        .replace(/·/g, "*")
        .replace(/\s+/g, "")
        .trim();
}

const PREFIXES = [
    { symbol: "p", factor: 1e-12 },
    { symbol: "n", factor: 1e-9 },
    { symbol: "u", factor: 1e-6 },
    { symbol: "m", factor: 1e-3 },
    { symbol: "c", factor: 1e-2 },
    { symbol: "k", factor: 1e3 },
    { symbol: "M", factor: 1e6 },
    { symbol: "G", factor: 1e9 }
];

const EXACT_UNIT_MAP = {
    "m": { base: "m", factor: 1 },
    "cm": { base: "m", factor: 1e-2 },
    "km": { base: "m", factor: 1e3 },

    "g": { base: "kg", factor: 1e-3 },
    "kg": { base: "kg", factor: 1 },

    "s": { base: "s", factor: 1 },
    "min": { base: "s", factor: 60 },
    "hr": { base: "s", factor: 3600 },

    "Hz": { base: "Hz", factor: 1 },
    "hz": { base: "Hz", factor: 1 },
    "khz": { base: "Hz", factor: 1e3 },
    "mhz": { base: "Hz", factor: 1e6 },
    "ghz": { base: "Hz", factor: 1e9 },

    "V": { base: "V", factor: 1 },
    "v": { base: "V", factor: 1 },
    "A": { base: "A", factor: 1 },
    "a": { base: "A", factor: 1 },
    "W": { base: "W", factor: 1 },
    "w": { base: "W", factor: 1 },
    "J": { base: "J", factor: 1 },
    "j": { base: "J", factor: 1 },
    "N": { base: "N", factor: 1 },
    "n": { base: "N", factor: 1 },
    "C": { base: "C", factor: 1 },
    "c": { base: "C", factor: 1 },
    "F": { base: "F", factor: 1 },
    "f": { base: "F", factor: 1 },
    "H": { base: "H", factor: 1 },
    "h": { base: "H", factor: 1 },
    "ohm": { base: "ohm", factor: 1 },
    "Ohm": { base: "ohm", factor: 1 },
    "OHM": { base: "ohm", factor: 1 },
    "Pa": { base: "Pa", factor: 1 },
    "pa": { base: "Pa", factor: 1 },
    "T": { base: "T", factor: 1 },
    "t": { base: "T", factor: 1 },

    "eV": { base: "J", factor: 1.602176634e-19 },
    "ev": { base: "J", factor: 1.602176634e-19 },

    "deg": { base: "deg", factor: 1 },
    "°": { base: "deg", factor: 1 },
    "rad": { base: "rad", factor: 1 },
    "%": { base: "ratio", factor: 0.01 }
};

const PREFIXABLE_BASE_UNITS = {
    "m": { base: "m", factor: 1 },
    "g": { base: "kg", factor: 1e-3 },
    "s": { base: "s", factor: 1 },
    "Hz": { base: "Hz", factor: 1 },
    "hz": { base: "Hz", factor: 1 },
    "V": { base: "V", factor: 1 },
    "v": { base: "V", factor: 1 },
    "A": { base: "A", factor: 1 },
    "a": { base: "A", factor: 1 },
    "W": { base: "W", factor: 1 },
    "w": { base: "W", factor: 1 },
    "J": { base: "J", factor: 1 },
    "j": { base: "J", factor: 1 },
    "N": { base: "N", factor: 1 },
    "n": { base: "N", factor: 1 },
    "C": { base: "C", factor: 1 },
    "c": { base: "C", factor: 1 },
    "F": { base: "F", factor: 1 },
    "f": { base: "F", factor: 1 },
    "H": { base: "H", factor: 1 },
    "h": { base: "H", factor: 1 },
    "ohm": { base: "ohm", factor: 1 },
    "Ohm": { base: "ohm", factor: 1 },
    "OHM": { base: "ohm", factor: 1 },
    "Pa": { base: "Pa", factor: 1 },
    "pa": { base: "Pa", factor: 1 },
    "T": { base: "T", factor: 1 },
    "t": { base: "T", factor: 1 },
    "eV": { base: "J", factor: 1.602176634e-19 },
    "ev": { base: "J", factor: 1.602176634e-19 }
};

function addExponentToBase(base, exponent) {
    if (exponent === 1) return base;
    return `${base}^${exponent}`;
}

function getSimpleUnitInfo(unitToken) {
    if (!unitToken) return null;

    const token = String(unitToken).trim();
    if (!token) return null;

    const exponentMatch = token.match(/^(.+?)\^([-+]?\d+)$/);
    if (exponentMatch) {
        const baseInfo = getSimpleUnitInfo(exponentMatch[1]);
        const exponent = Number(exponentMatch[2]);

        if (baseInfo && Number.isFinite(exponent)) {
            return {
                base: addExponentToBase(baseInfo.base, exponent),
                factor: Math.pow(baseInfo.factor, exponent),
                recognized: true
            };
        }
    }

    if (EXACT_UNIT_MAP[token]) {
        return {
            base: EXACT_UNIT_MAP[token].base,
            factor: EXACT_UNIT_MAP[token].factor,
            recognized: true
        };
    }

    for (const prefix of PREFIXES) {
        if (!token.startsWith(prefix.symbol) || token.length <= prefix.symbol.length) continue;

        const rest = token.slice(prefix.symbol.length);
        const baseInfo = PREFIXABLE_BASE_UNITS[rest];

        if (baseInfo) {
            return {
                base: baseInfo.base,
                factor: prefix.factor * baseInfo.factor,
                recognized: true
            };
        }
    }

    return null;
}

function normalizeCompoundBase(numeratorBases, denominatorBases) {
    if (
        numeratorBases.length === 1 &&
        denominatorBases.length === 1 &&
        ((numeratorBases[0] === "N" && denominatorBases[0] === "C") ||
            (numeratorBases[0] === "V" && denominatorBases[0] === "m"))
    ) {
        return "electric_field";
    }

    const numerator = numeratorBases.length ? numeratorBases.join("*") : "1";
    const denominator = denominatorBases.length ? denominatorBases.join("*") : "";

    return denominator ? `${numerator}/${denominator}` : numerator;
}

function getCompoundUnitInfo(unitOriginal) {
    if (!/[/*]/.test(unitOriginal)) return null;

    const parts = unitOriginal.split(/([/*])/).filter(part => part !== "");
    if (parts.length < 3) return null;

    let op = "*";
    let factor = 1;
    const numeratorBases = [];
    const denominatorBases = [];

    for (const part of parts) {
        if (part === "*") {
            op = "*";
            continue;
        }

        if (part === "/") {
            op = "/";
            continue;
        }

        const info = getSimpleUnitInfo(part);
        if (!info || !info.recognized) return null;

        if (op === "/") {
            factor /= info.factor;
            denominatorBases.push(info.base);
        } else {
            factor *= info.factor;
            numeratorBases.push(info.base);
        }
    }

    return {
        base: normalizeCompoundBase(numeratorBases, denominatorBases),
        factor,
        recognized: true
    };
}

function getUnitInfo(rawUnit) {
    const unitOriginal = normalizeUnitText(rawUnit);

    if (!unitOriginal) {
        return {
            base: null,
            factor: 1,
            recognized: false,
            unit: ""
        };
    }

    const simpleInfo = getSimpleUnitInfo(unitOriginal);
    if (simpleInfo) {
        return {
            base: simpleInfo.base,
            factor: simpleInfo.factor,
            recognized: true,
            unit: unitOriginal
        };
    }

    const compoundInfo = getCompoundUnitInfo(unitOriginal);
    if (compoundInfo) {
        return {
            base: compoundInfo.base,
            factor: compoundInfo.factor,
            recognized: true,
            unit: unitOriginal
        };
    }

    return {
        base: unitOriginal,
        factor: 1,
        recognized: false,
        unit: unitOriginal
    };
}

function extractUnitAfter(text, endIndex) {
    let rest = text.slice(endIndex).trim();

    rest = rest
        .replace(/^[\s*),.;:]+/g, "")
        .trim();

    const unitMatch = rest.match(
        /^([a-zA-ZµμΩΩ°%\\/\^\*·0-9+-]+(?:\s*(?:\/|\*|·)\s*[a-zA-ZµμΩΩ°%\\\^0-9+-]+)*)/
    );

    return unitMatch ? unitMatch[1] : "";
}

function parseMathNumber(text) {
    const numberPattern = "[-+]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[eE][-+]?\\d+)?";

    let regex = new RegExp(
        `(${numberPattern})\\s*\\\\?sqrt\\s*\\{?\\s*(${numberPattern})\\s*\\}?\\s*(?:\\*\\s*10\\s*\\^?\\s*([-+]?\\d+))?`,
        "i"
    );

    let match = text.match(regex);

    if (match) {
        const coefficient = Number(match[1]);
        const sqrtValue = Number(match[2]);
        const exponent = match[3] !== undefined ? Number(match[3]) : 0;

        if (
            Number.isFinite(coefficient) &&
            Number.isFinite(sqrtValue) &&
            Number.isFinite(exponent)
        ) {
            return {
                value: coefficient * Math.sqrt(sqrtValue) * Math.pow(10, exponent),
                start: match.index,
                end: match.index + match[0].length,
                raw: match[0]
            };
        }
    }

    regex = new RegExp(
        `\\\\?sqrt\\s*\\{?\\s*(${numberPattern})\\s*\\}?\\s*(?:\\*\\s*10\\s*\\^?\\s*([-+]?\\d+))?`,
        "i"
    );

    match = text.match(regex);

    if (match) {
        const sqrtValue = Number(match[1]);
        const exponent = match[2] !== undefined ? Number(match[2]) : 0;

        if (Number.isFinite(sqrtValue) && Number.isFinite(exponent)) {
            return {
                value: Math.sqrt(sqrtValue) * Math.pow(10, exponent),
                start: match.index,
                end: match.index + match[0].length,
                raw: match[0]
            };
        }
    }

    regex = new RegExp(
        `(${numberPattern})\\s*\\*\\s*10\\s*\\^?\\s*([-+]?\\d+)`,
        "i"
    );

    match = text.match(regex);

    if (match) {
        const coefficient = Number(match[1]);
        const exponent = Number(match[2]);

        if (Number.isFinite(coefficient) && Number.isFinite(exponent)) {
            return {
                value: coefficient * Math.pow(10, exponent),
                start: match.index,
                end: match.index + match[0].length,
                raw: match[0]
            };
        }
    }

    regex = new RegExp(`(${numberPattern})`, "i");

    match = text.match(regex);

    if (match) {
        const value = Number(match[1]);

        if (Number.isFinite(value)) {
            return {
                value,
                start: match.index,
                end: match.index + match[0].length,
                raw: match[0]
            };
        }
    }

    return null;
}

function extractQuantity(value) {
    const text = normalizeMathText(value);

    const parsed = parseMathNumber(text);
    if (!parsed) return null;

    const rawNumber = parsed.value;
    const rawUnit = extractUnitAfter(text, parsed.end);

    const unitInfo = getUnitInfo(rawUnit);
    const normalizedNumber = rawNumber * unitInfo.factor;

    return {
        raw_number: rawNumber,
        raw_unit: rawUnit,
        normalized_number: normalizedNumber,
        base_unit: unitInfo.base,
        unit_recognized: unitInfo.recognized,
        normalized_unit_label: unitInfo.base || null,
        parsed_raw_expression: parsed.raw
    };
}

function withinTolerance(goldValue, apiValue) {
    let tolerance = Math.abs(apiValue) * REL_TOLERANCE;
    tolerance = Math.max(tolerance, ABS_TOLERANCE_FLOOR);

    const lower = apiValue - tolerance;
    const upper = apiValue + tolerance;

    return {
        isCorrect: goldValue >= lower && goldValue <= upper,
        lower,
        upper,
        tolerance
    };
}

function compareQuantities(goldQ, apiQ) {
    if (goldQ === null) {
        return {
            isValid: false,
            isCorrect: false,
            errorReason: "Cannot parse gold answer number",
            lower: null,
            upper: null,
            tolerance: null,
            unitMismatch: false
        };
    }

    if (apiQ === null) {
        return {
            isValid: false,
            isCorrect: false,
            errorReason: "Cannot parse API answer number",
            lower: null,
            upper: null,
            tolerance: null,
            unitMismatch: false
        };
    }

    const bothRecognized = goldQ.unit_recognized && apiQ.unit_recognized;
    const bothHaveBase = goldQ.base_unit && apiQ.base_unit;

    if (bothRecognized && bothHaveBase && goldQ.base_unit !== apiQ.base_unit) {
        return {
            isValid: true,
            isCorrect: false,
            errorReason: `Unit mismatch: gold unit ${goldQ.base_unit}, API unit ${apiQ.base_unit}`,
            lower: null,
            upper: null,
            tolerance: null,
            unitMismatch: true,
            compared_gold_value: goldQ.normalized_number,
            compared_api_value: apiQ.normalized_number,
            used_normalized_units: true,
            compare_strategy: "unit_mismatch"
        };
    }

    const candidates = [];

    if (bothRecognized && goldQ.base_unit === apiQ.base_unit) {
        candidates.push({
            strategy: "normalized_same_unit",
            goldValue: goldQ.normalized_number,
            apiValue: apiQ.normalized_number,
            usedNormalized: true
        });
    } else {
        if (goldQ.unit_recognized) {
            candidates.push({
                strategy: "gold_normalized_vs_api_raw",
                goldValue: goldQ.normalized_number,
                apiValue: apiQ.raw_number,
                usedNormalized: true
            });
        }

        if (apiQ.unit_recognized) {
            candidates.push({
                strategy: "gold_raw_vs_api_normalized",
                goldValue: goldQ.raw_number,
                apiValue: apiQ.normalized_number,
                usedNormalized: true
            });
        }

        candidates.push({
            strategy: "raw_numeric",
            goldValue: goldQ.raw_number,
            apiValue: apiQ.raw_number,
            usedNormalized: false
        });
    }

    let firstResult = null;

    for (const candidate of candidates) {
        const result = withinTolerance(candidate.goldValue, candidate.apiValue);

        const enriched = {
            isValid: true,
            isCorrect: result.isCorrect,
            errorReason: result.isCorrect
                ? null
                : `Value outside tolerance: gold=${candidate.goldValue}, api=${candidate.apiValue}`,
            lower: result.lower,
            upper: result.upper,
            tolerance: result.tolerance,
            unitMismatch: false,
            compared_gold_value: candidate.goldValue,
            compared_api_value: candidate.apiValue,
            used_normalized_units: candidate.usedNormalized,
            compare_strategy: candidate.strategy
        };

        if (!firstResult) firstResult = enriched;
        if (result.isCorrect) return enriched;
    }

    return firstResult;
}

function normalizeForStringMatch(value) {
    return normalizeSuperscript(valueToText(value))
        .replace(/\\Omega/g, "ohm")
        .replace(/Ω|Ω/g, "ohm")
        .replace(/ω/g, "ohm")
        .replace(/\\mu/g, "u")
        .replace(/μ|µ/g, "u")
        .replace(/[`*_~]/g, " ")
        .replace(/["'“”‘’]/g, "")
        .replace(/\s+/g, " ")
        .trim()
        .toLowerCase();
}

function compareStrings(goldRaw, apiRaw) {
    const goldText = normalizeForStringMatch(goldRaw);
    const apiText = normalizeForStringMatch(apiRaw);

    if (!goldText) {
        return {
            mode: "string_contains",
            isValid: false,
            isCorrect: false,
            errorReason: "Gold answer is empty, cannot compare as string",
            normalizedGoldString: goldText,
            normalizedApiString: apiText
        };
    }

    if (!apiText) {
        return {
            mode: "string_contains",
            isValid: false,
            isCorrect: false,
            errorReason: "API answer is empty, cannot compare as string",
            normalizedGoldString: goldText,
            normalizedApiString: apiText
        };
    }

    const isCorrect = goldText.includes(apiText);

    return {
        mode: "string_contains",
        isValid: true,
        isCorrect,
        errorReason: isCorrect
            ? null
            : `String mismatch: API answer is not contained in gold answer`,
        normalizedGoldString: goldText,
        normalizedApiString: apiText,
        compare_strategy: "api_string_in_gold_string"
    };
}

function compareAnswers(goldRaw, apiRaw) {
    const goldBool = extractBoolean(goldRaw);
    const apiBool = extractBoolean(apiRaw);

    if (goldBool !== null) {
        if (apiBool === null) {
            return {
                mode: "boolean",
                isValid: false,
                isCorrect: false,
                errorReason: "Gold answer is Yes/No, but API answer is not parseable as Yes/No",
                goldBool,
                apiBool
            };
        }

        return {
            mode: "boolean",
            isValid: true,
            isCorrect: goldBool === apiBool,
            errorReason: goldBool === apiBool
                ? null
                : `Boolean mismatch: gold=${goldBool}, api=${apiBool}`,
            goldBool,
            apiBool
        };
    }

    const goldQ = extractQuantity(goldRaw);
    const apiQ = extractQuantity(apiRaw);

    if (goldQ !== null || apiQ !== null) {
        const result = compareQuantities(goldQ, apiQ);

        return {
            mode: "numeric_unit",
            ...result,
            goldQuantity: goldQ,
            apiQuantity: apiQ
        };
    }

    return compareStrings(goldRaw, apiRaw);
}

// -------------------------------
// Main logic
// -------------------------------

const id =
    pm.variables.get("current_id") ||
    pm.iterationData.get("id") ||
    `ITER_${pm.info.iteration + 1}`;

const question =
    pm.variables.get("current_question") ||
    valueToText(pm.iterationData.get("question")).trim();

const goldAnswerRaw =
    pm.variables.get("current_gold_answer_full") ||
    combineAnswerAndUnit(
        pm.iterationData.get("answer"),
        pm.iterationData.get("unit")
    );

let apiJson = null;
let apiAnswerRaw = null;
let responseParseError = null;

let totalCount = getCollectionNumber("total_count") + 1;
let correctCount = getCollectionNumber("correct_count");
let wrongCount = getCollectionNumber("wrong_count");
let invalidCount = getCollectionNumber("invalid_count");

setCollectionNumber("total_count", totalCount);

try {
    apiJson = pm.response.json();
    apiAnswerRaw = apiJson.answer;
} catch (e) {
    responseParseError = e.message || String(e);
    apiJson = null;
    apiAnswerRaw = null;
}

let comparison = null;

if (responseParseError) {
    comparison = {
        mode: "response_json",
        isValid: false,
        isCorrect: false,
        errorReason: `Cannot parse API response JSON: ${responseParseError}`
    };
} else if (
    apiAnswerRaw === undefined ||
    apiAnswerRaw === null ||
    valueToText(apiAnswerRaw).trim() === ""
) {
    comparison = {
        mode: "missing_answer",
        isValid: false,
        isCorrect: false,
        errorReason: "API response does not contain field `answer`"
    };
} else {
    comparison = compareAnswers(goldAnswerRaw, apiAnswerRaw);
}

const statusOk = pm.response.code === 200;
const isValid = comparison.isValid;
const isCorrect = comparison.isCorrect;
const overallPass = statusOk && isValid && isCorrect;

if (!statusOk || !isValid) {
    invalidCount += 1;
    setCollectionNumber("invalid_count", invalidCount);
} else if (overallPass) {
    correctCount += 1;
    setCollectionNumber("correct_count", correctCount);
} else {
    wrongCount += 1;
    setCollectionNumber("wrong_count", wrongCount);
}

if (!overallPass) {
    let wrongCases = [];

    try {
        wrongCases = JSON.parse(pm.collectionVariables.get("wrong_cases") || "[]");
    } catch (e) {
        wrongCases = [];
    }

    if (wrongCases.length < MAX_WRONG_CASES) {
        wrongCases.push({
            iteration: pm.info.iteration + 1,
            id,
            question,
            status_code: pm.response.code,
            status_ok: statusOk,
            overall_pass: overallPass,
            gold_answer_raw: goldAnswerRaw,
            api_answer_raw: apiAnswerRaw,
            mode: comparison.mode,
            is_valid: isValid,
            is_correct: isCorrect,
            error_reason: comparison.errorReason,

            gold_bool: comparison.goldBool,
            api_bool: comparison.apiBool,

            gold_quantity: comparison.goldQuantity,
            api_quantity: comparison.apiQuantity,
            compared_gold_value: comparison.compared_gold_value,
            compared_api_value: comparison.compared_api_value,
            lower_bound: comparison.lower,
            upper_bound: comparison.upper,
            tolerance: comparison.tolerance,
            used_normalized_units: comparison.used_normalized_units,
            compare_strategy: comparison.compare_strategy,
            unit_mismatch: comparison.unitMismatch,

            normalized_gold_string: comparison.normalizedGoldString,
            normalized_api_string: comparison.normalizedApiString,

            response_body: pm.response.text()
        });

        pm.collectionVariables.set("wrong_cases", JSON.stringify(wrongCases, null, 2));
    }
}

pm.test(`[${id}] Overall result`, function () {
    pm.expect(
        statusOk,
        `Status code must be 200, got ${pm.response.code}. Response: ${pm.response.text()}`
    ).to.eql(true);

    pm.expect(
        isValid,
        comparison.errorReason || "Gold answer and API answer must be comparable"
    ).to.eql(true);

    pm.expect(
        isCorrect,
        JSON.stringify({
            mode: comparison.mode,
            gold: goldAnswerRaw,
            api: apiAnswerRaw,
            reason: comparison.errorReason,
            compared_gold_value: comparison.compared_gold_value,
            compared_api_value: comparison.compared_api_value,
            range: [comparison.lower, comparison.upper],
            tolerance: comparison.tolerance,
            used_normalized_units: comparison.used_normalized_units,
            compare_strategy: comparison.compare_strategy,
            normalized_gold_string: comparison.normalizedGoldString,
            normalized_api_string: comparison.normalizedApiString
        })
    ).to.eql(true);
});

console.log("========== POST-REQUEST RESULT ==========");
console.log({
    iteration: pm.info.iteration + 1,
    id,
    status_code: pm.response.code,
    status_ok: statusOk,
    overall_pass: overallPass,
    gold_answer_raw: goldAnswerRaw,
    api_answer_raw: apiAnswerRaw,
    mode: comparison.mode,
    is_correct: isCorrect,
    is_valid: isValid,
    error_reason: comparison.errorReason,
    gold_quantity: comparison.goldQuantity,
    api_quantity: comparison.apiQuantity,
    compared_gold_value: comparison.compared_gold_value,
    compared_api_value: comparison.compared_api_value,
    lower_bound: comparison.lower,
    upper_bound: comparison.upper,
    tolerance: comparison.tolerance,
    used_normalized_units: comparison.used_normalized_units,
    compare_strategy: comparison.compare_strategy,
    gold_bool: comparison.goldBool,
    api_bool: comparison.apiBool,
    normalized_gold_string: comparison.normalizedGoldString,
    normalized_api_string: comparison.normalizedApiString
});

if (pm.info.iteration === pm.info.iterationCount - 1) {
    const finalTotal = getCollectionNumber("total_count");
    const finalCorrect = getCollectionNumber("correct_count");
    const finalWrong = getCollectionNumber("wrong_count");
    const finalInvalid = getCollectionNumber("invalid_count");
    const accuracy = finalTotal > 0
        ? (finalCorrect / finalTotal * 100).toFixed(2)
        : "0.00";

    console.log("========== FINAL RESULT ==========");
    console.log(`CORRECT: ${finalCorrect}/${finalTotal}`);
    console.log(`WRONG: ${finalWrong}`);
    console.log(`INVALID: ${finalInvalid}`);
    console.log(`ACCURACY: ${accuracy}%`);
    console.log("FIRST WRONG/INVALID CASES:");
    console.log(JSON.parse(pm.collectionVariables.get("wrong_cases") || "[]"));
}