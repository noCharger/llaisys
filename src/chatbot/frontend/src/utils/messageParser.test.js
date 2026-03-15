
import { describe, it, expect } from 'vitest';
import { parseMessage } from './messageParser';

describe('messageParser', () => {
    it('should parse simple text', () => {
        const input = "Hello world";
        const result = parseMessage(input);
        expect(result).toEqual([{ type: 'text', content: "Hello world" }]);
    });

    it('should parse think block followed by text', () => {
        const input = "<think>This is a thought</think>Hello there";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'think', content: "This is a thought" },
            { type: 'text', content: "Hello there" }
        ]);
    });

    it('should parse text followed by think block', () => {
        const input = "Prefix<think>Thought</think>";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'text', content: "Prefix" },
            { type: 'think', content: "Thought" }
        ]);
    });

    it('should treat unclosed think tag as open-ended think block', () => {
        const input = "Start <think>Unclosed thought";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'text', content: "Start " },
            { type: 'think', content: "Unclosed thought" }
        ]);
    });

    it('should handle missing opening think tag at the start', () => {
        const input = "Implicit thought</think>Response";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'think', content: "Implicit thought" },
            { type: 'text', content: "Response" }
        ]);
    });

    it('should handle multiple think blocks', () => {
        const input = "<think>One</think>Middle<think>Two</think>End";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'think', content: "One" },
            { type: 'text', content: "Middle" },
            { type: 'think', content: "Two" },
            { type: 'text', content: "End" }
        ]);
    });

    it('should handle empty content', () => {
        expect(parseMessage("")).toEqual([]);
        expect(parseMessage(null)).toEqual([]);
    });

    it('should handle multiline think blocks', () => {
        const input = "<think>\nLine 1\nLine 2\n</think>\nResponse";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'think', content: "\nLine 1\nLine 2\n" },
            { type: 'text', content: "\nResponse" }
        ]);
    });

    it('should preserve whitespace in text', () => {
        const input = "Hello <think>...</think> World";
        const result = parseMessage(input);
        expect(result).toEqual([
            { type: 'text', content: "Hello " },
            { type: 'think', content: "..." },
            { type: 'text', content: " World" }
        ]);
    });
});
