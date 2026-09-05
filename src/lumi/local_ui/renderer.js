"use strict";

(function initializeLumiRenderer(globalScope) {
  const VERSION = "lumi-safe-markdown-math-v0.1";
  const MAX_INLINE_DEPTH = 4;
  const MAX_MATH_CHARACTERS = 8000;
  const FENCE = /^ {0,3}```([A-Za-z0-9_+.-]*)[ \t]*$/;
  const HEADING = /^ {0,3}(#{1,4})[ \t]+(.+)$/;
  const ORDERED_ITEM = /^ {0,3}(\d+)[.)][ \t]+(.+)$/;
  const UNORDERED_ITEM = /^ {0,3}[-+*][ \t]+(.+)$/;
  const BLOCKQUOTE = /^ {0,3}>[ \t]?(.*)$/;
  const HORIZONTAL_RULE = /^ {0,3}(?:-{3,}|_{3,}|\*{3,})[ \t]*$/;

  function pushText(tokens, value) {
    if (!value) return;
    const previous = tokens[tokens.length - 1];
    if (previous?.type === "text") {
      previous.value += value;
    } else {
      tokens.push({ type: "text", value });
    }
  }

  function findClosing(source, marker, start) {
    let cursor = start;
    while (cursor < source.length) {
      const found = source.indexOf(marker, cursor);
      if (found === -1) return -1;
      let backslashes = 0;
      for (let index = found - 1; index >= 0 && source[index] === "\\"; index -= 1) {
        backslashes += 1;
      }
      if (backslashes % 2 === 0) return found;
      cursor = found + marker.length;
    }
    return -1;
  }

  function isSafeLink(value) {
    if (typeof value !== "string" || !value.trim()) return false;
    try {
      const parsed = new URL(value.trim());
      return ["https:", "http:", "mailto:"].includes(parsed.protocol);
    } catch {
      return false;
    }
  }

  function tokenizeInline(source, depth = 0) {
    const input = String(source ?? "");
    const tokens = [];
    let index = 0;
    while (index < input.length) {
      const character = input[index];

      if (character === "\n") {
        tokens.push({ type: "line_break" });
        index += 1;
        continue;
      }

      if (character === "\\" && index + 1 < input.length) {
        if (input[index + 1] === "(") {
          const end = findClosing(input, "\\)", index + 2);
          if (end !== -1 && end > index + 2) {
            tokens.push({
              display: false,
              type: "math",
              value: input.slice(index + 2, end),
            });
            index = end + 2;
            continue;
          }
        }
        if ("\\`*_[]()$".includes(input[index + 1])) {
          pushText(tokens, input[index + 1]);
          index += 2;
          continue;
        }
      }

      if (character === "`") {
        const end = findClosing(input, "`", index + 1);
        if (end !== -1) {
          tokens.push({ type: "code", value: input.slice(index + 1, end) });
          index = end + 1;
          continue;
        }
      }

      if (character === "$" && input[index + 1] !== "$") {
        const end = findClosing(input, "$", index + 1);
        const value = end === -1 ? "" : input.slice(index + 1, end);
        const looksLikeTwoCurrencyAmounts =
          /^\d/.test(value) && /\s$/.test(value) && /\d/.test(input[end + 1] || "");
        if (
          end !== -1 &&
          value.length > 0 &&
          value.trim() === value &&
          !looksLikeTwoCurrencyAmounts
        ) {
          tokens.push({ display: false, type: "math", value });
          index = end + 1;
          continue;
        }
      }

      if (depth < MAX_INLINE_DEPTH && input.startsWith("**", index)) {
        const end = findClosing(input, "**", index + 2);
        if (end > index + 2) {
          tokens.push({
            children: tokenizeInline(input.slice(index + 2, end), depth + 1),
            type: "strong",
          });
          index = end + 2;
          continue;
        }
      }

      if (depth < MAX_INLINE_DEPTH && character === "*") {
        const end = findClosing(input, "*", index + 1);
        if (end > index + 1) {
          tokens.push({
            children: tokenizeInline(input.slice(index + 1, end), depth + 1),
            type: "emphasis",
          });
          index = end + 1;
          continue;
        }
      }

      if (depth < MAX_INLINE_DEPTH && character === "[") {
        const labelEnd = findClosing(input, "]", index + 1);
        if (labelEnd !== -1 && input[labelEnd + 1] === "(") {
          const urlEnd = findClosing(input, ")", labelEnd + 2);
          const href = urlEnd === -1 ? "" : input.slice(labelEnd + 2, urlEnd).trim();
          if (urlEnd !== -1 && isSafeLink(href)) {
            tokens.push({
              children: tokenizeInline(input.slice(index + 1, labelEnd), depth + 1),
              href,
              type: "link",
            });
            index = urlEnd + 1;
            continue;
          }
        }
      }

      pushText(tokens, character);
      index += 1;
    }
    return tokens;
  }

  function displayMathStart(line) {
    const trimmed = line.trim();
    if (trimmed === "$$") return { close: "$$", offset: 2 };
    if (trimmed === "\\[") return { close: "\\]", offset: 2 };
    if (trimmed.startsWith("$$") && trimmed.endsWith("$$") && trimmed.length > 4) {
      return { close: "$$", inline: trimmed.slice(2, -2) };
    }
    if (trimmed.startsWith("\\[") && trimmed.endsWith("\\]") && trimmed.length > 4) {
      return { close: "\\]", inline: trimmed.slice(2, -2) };
    }
    return null;
  }

  function lineStartsBlock(line) {
    return (
      FENCE.test(line) ||
      HEADING.test(line) ||
      ORDERED_ITEM.test(line) ||
      UNORDERED_ITEM.test(line) ||
      BLOCKQUOTE.test(line) ||
      HORIZONTAL_RULE.test(line) ||
      displayMathStart(line) !== null
    );
  }

  function tokenizeBlocks(source) {
    const lines = String(source ?? "").replace(/\r\n?/g, "\n").split("\n");
    const tokens = [];
    let index = 0;
    while (index < lines.length) {
      const line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      const fence = line.match(FENCE);
      if (fence) {
        const code = [];
        let closed = false;
        index += 1;
        while (index < lines.length) {
          if (/^ {0,3}```[ \t]*$/.test(lines[index])) {
            closed = true;
            index += 1;
            break;
          }
          code.push(lines[index]);
          index += 1;
        }
        tokens.push({
          closed,
          language: fence[1] || "",
          type: "code_block",
          value: code.join("\n"),
        });
        continue;
      }

      const display = displayMathStart(line);
      if (display) {
        if (typeof display.inline === "string") {
          tokens.push({ type: "math_block", value: display.inline });
          index += 1;
          continue;
        }
        const math = [];
        let closed = false;
        index += 1;
        while (index < lines.length) {
          if (lines[index].trim() === display.close) {
            closed = true;
            index += 1;
            break;
          }
          math.push(lines[index]);
          index += 1;
        }
        tokens.push({
          closed,
          delimiter: display.close === "$$" ? "$$" : "\\[",
          type: "math_block",
          value: math.join("\n"),
        });
        continue;
      }

      const heading = line.match(HEADING);
      if (heading) {
        tokens.push({ level: heading[1].length, type: "heading", value: heading[2] });
        index += 1;
        continue;
      }

      if (HORIZONTAL_RULE.test(line)) {
        tokens.push({ type: "horizontal_rule" });
        index += 1;
        continue;
      }

      const quote = line.match(BLOCKQUOTE);
      if (quote) {
        const values = [quote[1]];
        index += 1;
        while (index < lines.length) {
          const continuation = lines[index].match(BLOCKQUOTE);
          if (!continuation) break;
          values.push(continuation[1]);
          index += 1;
        }
        tokens.push({ type: "blockquote", value: values.join("\n") });
        continue;
      }

      const ordered = line.match(ORDERED_ITEM);
      const unordered = line.match(UNORDERED_ITEM);
      if (ordered || unordered) {
        const orderedList = Boolean(ordered);
        const matcher = orderedList ? ORDERED_ITEM : UNORDERED_ITEM;
        const items = [];
        const start = ordered ? Number(ordered[1]) : null;
        while (index < lines.length) {
          const item = lines[index].match(matcher);
          if (!item) break;
          items.push(item[orderedList ? 2 : 1]);
          index += 1;
        }
        tokens.push({ items, ordered: orderedList, start, type: "list" });
        continue;
      }

      const paragraph = [line];
      index += 1;
      while (
        index < lines.length &&
        lines[index].trim() &&
        !lineStartsBlock(lines[index])
      ) {
        paragraph.push(lines[index]);
        index += 1;
      }
      tokens.push({ type: "paragraph", value: paragraph.join("\n") });
    }
    return tokens;
  }

  function renderMath(element, value, displayMode) {
    const expression = String(value ?? "");
    const katex = globalScope.katex;
    if (!katex || expression.length > MAX_MATH_CHARACTERS || !expression.trim()) {
      element.classList.add("math-fallback");
      element.textContent = displayMode ? `$$${expression}$$` : `$${expression}$`;
      return;
    }
    try {
      katex.render(expression, element, {
        displayMode,
        maxExpand: 1000,
        maxSize: 20,
        output: "htmlAndMathml",
        strict: "warn",
        throwOnError: true,
        trust: false,
      });
      element.classList.add("math-rendered");
    } catch {
      element.classList.add("math-fallback");
      element.textContent = displayMode ? `$$${expression}$$` : `$${expression}$`;
    }
  }

  function appendInline(parent, tokens, documentRef) {
    for (const token of tokens) {
      if (token.type === "text") {
        parent.append(documentRef.createTextNode(token.value));
      } else if (token.type === "line_break") {
        parent.append(documentRef.createElement("br"));
      } else if (token.type === "code") {
        const code = documentRef.createElement("code");
        code.textContent = token.value;
        parent.append(code);
      } else if (token.type === "strong" || token.type === "emphasis") {
        const child = documentRef.createElement(token.type === "strong" ? "strong" : "em");
        appendInline(child, token.children, documentRef);
        parent.append(child);
      } else if (token.type === "link") {
        const link = documentRef.createElement("a");
        link.href = token.href;
        link.rel = "nofollow noopener noreferrer";
        link.target = "_blank";
        appendInline(link, token.children, documentRef);
        parent.append(link);
      } else if (token.type === "math") {
        const math = documentRef.createElement("span");
        math.className = "math-inline";
        renderMath(math, token.value, false);
        parent.append(math);
      }
    }
  }

  function renderMessage(container, source) {
    const documentRef = container.ownerDocument || globalScope.document;
    const fragment = documentRef.createDocumentFragment();
    for (const token of tokenizeBlocks(source)) {
      let block;
      if (token.type === "paragraph") {
        block = documentRef.createElement("p");
        appendInline(block, tokenizeInline(token.value), documentRef);
      } else if (token.type === "heading") {
        block = documentRef.createElement(`h${Math.min(token.level + 1, 5)}`);
        appendInline(block, tokenizeInline(token.value), documentRef);
      } else if (token.type === "code_block") {
        block = documentRef.createElement("pre");
        const code = documentRef.createElement("code");
        if (token.language) code.className = `language-${token.language}`;
        code.textContent = token.value;
        block.append(code);
      } else if (token.type === "math_block") {
        block = documentRef.createElement("div");
        block.className = "math-block";
        if (token.closed === false) {
          block.classList.add("math-fallback");
          const marker = token.delimiter || "$$";
          block.textContent = `${marker}\n${token.value}`;
        } else {
          renderMath(block, token.value, true);
        }
      } else if (token.type === "blockquote") {
        block = documentRef.createElement("blockquote");
        appendInline(block, tokenizeInline(token.value), documentRef);
      } else if (token.type === "list") {
        block = documentRef.createElement(token.ordered ? "ol" : "ul");
        if (token.ordered && token.start !== 1) block.start = token.start;
        for (const item of token.items) {
          const listItem = documentRef.createElement("li");
          appendInline(listItem, tokenizeInline(item), documentRef);
          block.append(listItem);
        }
      } else if (token.type === "horizontal_rule") {
        block = documentRef.createElement("hr");
      }
      if (block) fragment.append(block);
    }
    container.replaceChildren(fragment);
  }

  const renderer = { VERSION, isSafeLink, renderMessage, tokenizeBlocks, tokenizeInline };
  globalScope.LumiRenderer = renderer;
  if (typeof module !== "undefined" && module.exports) module.exports = renderer;
})(typeof window !== "undefined" ? window : globalThis);
