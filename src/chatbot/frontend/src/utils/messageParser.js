
/**
 * Parses a raw message string into structured segments.
 * 
 * @param {string} content - The raw message content
 * @returns {Array<{type: 'text'|'think', content: string}>} - Array of segments
 */
export const parseMessage = (content) => {
    if (!content && content !== '') return [];
    if (content === '') return [];
    
    const segments = [];
    let currentPos = 0;
    
    while (currentPos < content.length) {
        const startTagIndex = content.indexOf('<think>', currentPos);
        const endTagIndex = content.indexOf('</think>', currentPos);
        
        if (endTagIndex === -1) {
            if (startTagIndex !== -1) {
                // Case: Unclosed think block (streaming or truncated)
                if (startTagIndex > currentPos) {
                    const textContent = content.substring(currentPos, startTagIndex);
                    if (textContent) {
                        segments.push({ type: 'text', content: textContent });
                    }
                }
                
                const thoughtContent = content.substring(startTagIndex + 7);
                if (thoughtContent) {
                    segments.push({ type: 'think', content: thoughtContent });
                }
                break;
            }

            const remainingText = content.substring(currentPos);
            if (remainingText) {
                segments.push({ type: 'text', content: remainingText });
            }
            break;
        }
                
        if (startTagIndex === -1 || startTagIndex > endTagIndex) {            
            const thoughtContent = content.substring(currentPos, endTagIndex);
            if (thoughtContent) {
                segments.push({ type: 'think', content: thoughtContent });
            }
            
            currentPos = endTagIndex + 8;
        } else {            
            if (startTagIndex > currentPos) {
                const textContent = content.substring(currentPos, startTagIndex);
                if (textContent) {
                    segments.push({ type: 'text', content: textContent });
                }
            }
            
            const thoughtContent = content.substring(startTagIndex + 7, endTagIndex); // 7 is length of '<think>'
            if (thoughtContent) {
                 segments.push({ type: 'think', content: thoughtContent });
            }
            
            currentPos = endTagIndex + 8;
        }
    }
    
    return segments;
};
