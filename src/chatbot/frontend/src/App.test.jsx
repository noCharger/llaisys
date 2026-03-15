import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import App from './App';
import * as useChatModule from './hooks/useChat';

// Mock the hook
vi.mock('./hooks/useChat', () => ({
  __esModule: true,
  default: vi.fn(),
}));

describe('App Component', () => {
  it('renders navbar', () => {
    useChatModule.default.mockReturnValue({
        messages: [],
        sendMessage: vi.fn(),
        isThinking: false,
        error: null
    });
    render(<App />);
    expect(screen.getByText('LLAISYS AI Chat')).toBeInTheDocument();
  });

  it('handles user input', () => {
    const sendMessageMock = vi.fn();
    useChatModule.default.mockReturnValue({
        messages: [],
        sendMessage: sendMessageMock,
        isThinking: false,
        error: null
    });

    render(<App />);
    const input = screen.getByPlaceholderText('Type your message...');
    const button = screen.getByText('Send');

    fireEvent.change(input, { target: { value: 'Hello' } });
    fireEvent.click(button);

    expect(sendMessageMock).toHaveBeenCalledWith('Hello');
  });

  it('displays thinking indicator', () => {
    useChatModule.default.mockReturnValue({
        messages: [],
        sendMessage: vi.fn(),
        isThinking: true,
        error: null
    });

    render(<App />);
    expect(screen.getByText('Thinking')).toBeInTheDocument();
  });
});
