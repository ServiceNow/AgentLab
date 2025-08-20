"""JavaScript Event Listeners for Human Trace Capture

This module contains all the JavaScript code for capturing comprehensive
browser interactions including mouse, keyboard, form, scroll, and focus events.
"""


def get_interaction_tracking_script() -> str:
    """Get the complete JavaScript code for interaction tracking."""
    return (
        """
        window.__acted = false;
        window.__interactions = [];
        
        // Debug mode - set to true to see all events in console
        window.__debug_events = false; 
        
        function captureInteraction(type, event, extra = {}) {
            // Skip our own recording indicators
            if (event.target.id === '__rec' || event.target.id === '__rec_border' || 
                event.target.closest('#__rec') || event.target.closest('#__rec_border')) {
                return;
            }
            
            const interaction = {
                type: type,
                timestamp: Date.now(),
                coords: {
                    x: event.clientX || 0,
                    y: event.clientY || 0
                },
                target: {
                    tagName: event.target.tagName,
                    id: event.target.id || null,
                    className: event.target.className || null,
                    text: event.target.textContent?.slice(0, 50) || null,
                    bid: event.target.getAttribute('bid') || null
                },
                ...extra
            };
            
            window.__interactions.push(interaction);
            window.__acted = true;
            
            // Debug logging
            if (window.__debug_events) {
                console.log(`🎯 Captured: ${type}`, interaction);
            }
            
            // Update indicators immediately
            const indicator = document.getElementById('__rec');
            const border = document.getElementById('__rec_border');
            if (indicator) {
                indicator.innerHTML = '✅ ACTION DETECTED - SAVING...';
                indicator.style.background = '#28a745';
                indicator.style.animation = 'none';
            }
            if (border) {
                border.style.border = '8px solid #28a745';
                border.style.animation = 'none';
            }
        }
        
        // Debug function - add this temporarily to see what events fire
        if (window.__debug_events) {
            ['input', 'change', 'select', 'focus', 'click', 'keydown', 'paste', 'cut', 'copy'].forEach(eventType => {
                document.addEventListener(eventType, (e) => {
                    console.log(`🔍 DEBUG: ${eventType} on`, e.target.tagName, e.target.type, e.target);
                }, true);
            });
        }
        
        """
        + get_mouse_event_listeners()
        + """
        """
        + get_keyboard_event_listeners()
        + """
        """
        + get_form_event_listeners()
        + """
        """
        + get_scroll_event_listeners()
        + """
        """
        + get_focus_event_listeners()
        + """
        
        console.log('Comprehensive interaction tracking initialized');
    """
    )


def get_mouse_event_listeners() -> str:
    """Get JavaScript code for mouse event listeners."""
    return """
        // Mouse events with comprehensive button tracking and performance optimizations
        let lastClickTime = 0;
        
        document.addEventListener('click', (e) => {
            const now = Date.now();
            // Prevent spam clicking from creating too many events (minimum 50ms between clicks)
            if (now - lastClickTime < 50) return;
            lastClickTime = now;
            
            captureInteraction('click', e, {
                button: e.button, // 0=left, 1=middle, 2=right
                buttons: e.buttons, // bitmask of pressed buttons
                buttonName: ['left', 'middle', 'right'][e.button] || 'unknown',
                detail: e.detail, // click count (single, double, etc.)
                clickType: e.detail === 1 ? 'single' : e.detail === 2 ? 'double' : `${e.detail}x`
            });
        }, true);
        
        document.addEventListener('dblclick', (e) => {
            captureInteraction('dblclick', e, {
                button: e.button,
                buttonName: ['left', 'middle', 'right'][e.button] || 'unknown'
            });
        }, true);
        
        document.addEventListener('mousedown', (e) => {
            captureInteraction('mousedown', e, {
                button: e.button,
                buttons: e.buttons,
                buttonName: ['left', 'middle', 'right'][e.button] || 'unknown'
            });
        }, true);
        
        document.addEventListener('mouseup', (e) => {
            captureInteraction('mouseup', e, {
                button: e.button,
                buttons: e.buttons,
                buttonName: ['left', 'middle', 'right'][e.button] || 'unknown'
            });
        }, true);
        
        // Context menu (right-click menu)
        document.addEventListener('contextmenu', (e) => {
            captureInteraction('contextmenu', e, {
                button: e.button,
                buttonName: 'right'
            });
        }, true);
        
        // Middle mouse button events (often used for scrolling/opening in new tab)
        document.addEventListener('auxclick', (e) => {
            captureInteraction('auxclick', e, {
                button: e.button,
                buttonName: e.button === 1 ? 'middle' : (e.button === 2 ? 'right' : 'other'),
                detail: e.detail
            });
        }, true);
        
        // Enhanced drag tracking (without redundant mousedown)
        let isDragging = false;
        let dragStart = null;
        let dragButton = null;
        let hasDraggedSignificantly = false;
        
        document.addEventListener('mousedown', (e) => {
            isDragging = true;
            dragButton = e.button;
            hasDraggedSignificantly = false;
            dragStart = {
                x: e.clientX, 
                y: e.clientY, 
                time: Date.now(),
                button: e.button,
                buttonName: ['left', 'middle', 'right'][e.button] || 'unknown'
            };
        }, true);
        
        document.addEventListener('mousemove', (e) => {
            if (isDragging && dragStart) {
                const distance = Math.sqrt(
                    Math.pow(e.clientX - dragStart.x, 2) + 
                    Math.pow(e.clientY - dragStart.y, 2)
                );
                if (distance > 5 && !hasDraggedSignificantly) { 
                    // Only capture the start of a significant drag, not every movement
                    hasDraggedSignificantly = true;
                    captureInteraction('drag_start', e, {
                        startX: dragStart.x,
                        startY: dragStart.y,
                        endX: e.clientX,
                        endY: e.clientY,
                        distance: distance,
                        button: dragButton,
                        buttonName: dragStart.buttonName,
                        duration: Date.now() - dragStart.time
                    });
                }
            }
            // Note: Removed general mousemove tracking to reduce noise
        }, true);
        
        document.addEventListener('mouseup', (e) => {
            if (isDragging && dragStart && hasDraggedSignificantly) {
                const distance = Math.sqrt(
                    Math.pow(e.clientX - dragStart.x, 2) + 
                    Math.pow(e.clientY - dragStart.y, 2)
                );
                captureInteraction('drag_end', e, {
                    startX: dragStart.x,
                    startY: dragStart.y,
                    endX: e.clientX,
                    endY: e.clientY,
                    distance: distance,
                    duration: Date.now() - dragStart.time,
                    button: dragButton,
                    buttonName: dragStart.buttonName,
                    totalDistance: distance
                });
            }
            isDragging = false;
            dragStart = null;
            dragButton = null;
            hasDraggedSignificantly = false;
        }, true);
        
        // Drag and drop events
        document.addEventListener('dragstart', (e) => {
            captureInteraction('dragstart', e, {
                dataTransfer: {
                    effectAllowed: e.dataTransfer.effectAllowed,
                    types: Array.from(e.dataTransfer.types)
                }
            });
        }, true);
        
        document.addEventListener('dragend', (e) => {
            captureInteraction('dragend', e, {
                dataTransfer: {
                    dropEffect: e.dataTransfer.dropEffect
                }
            });
        }, true);
        
        document.addEventListener('drop', (e) => {
            captureInteraction('drop', e, {
                dataTransfer: {
                    dropEffect: e.dataTransfer.dropEffect,
                    types: Array.from(e.dataTransfer.types)
                },
                files: e.dataTransfer.files.length > 0 ? Array.from(e.dataTransfer.files).map(f => ({
                    name: f.name,
                    type: f.type,
                    size: f.size
                })) : null
            });
        }, true);
    """


def get_keyboard_event_listeners() -> str:
    """Get JavaScript code for keyboard event listeners."""
    return """
        // Keyboard events with shortcut detection
        document.addEventListener('keydown', (e) => {
            let shortcut = null;
            if (e.ctrlKey || e.metaKey) {
                const modifier = e.ctrlKey ? 'Ctrl' : 'Cmd';
                const key = e.key.length === 1 ? e.key.toUpperCase() : e.key;
                shortcut = `${modifier}+${key}`;
            } else if (e.altKey && e.key.length === 1) {
                shortcut = `Alt+${e.key.toUpperCase()}`;
            } else if (e.shiftKey && e.key.length === 1) {
                shortcut = `Shift+${e.key.toUpperCase()}`;
            }
            
            captureInteraction('keydown', e, {
                key: e.key,
                code: e.code,
                ctrlKey: e.ctrlKey,
                shiftKey: e.shiftKey,
                altKey: e.altKey,
                metaKey: e.metaKey,
                shortcut: shortcut
            });
        }, true);
        
        document.addEventListener('keyup', (e) => {
            captureInteraction('keyup', e, {
                key: e.key,
                code: e.code
            });
        }, true);
    """


def get_form_event_listeners() -> str:
    """Get JavaScript code for form event listeners."""
    return """
        // Input events with throttling to prevent spam during fast typing
        let inputTimeout;
        let lastInputValue = '';
        
        document.addEventListener('input', (e) => {
            if (['INPUT', 'TEXTAREA'].includes(e.target.tagName) || e.target.contentEditable === 'true') {
                clearTimeout(inputTimeout);
                inputTimeout = setTimeout(() => {
                    const currentValue = e.target.value || e.target.textContent;
                    // Only capture if value actually changed significantly
                    if (currentValue !== lastInputValue) {
                        lastInputValue = currentValue;
                        captureInteraction('input', e, {
                            value: currentValue,
                            inputType: e.inputType || null,
                            valueLength: currentValue.length
                        });
                    }
                }, 50); // Reduced from 300ms to 50ms for better responsiveness
            }
        }, true);
        
        // Immediate input capture (without throttling) for certain cases
        document.addEventListener('input', (e) => {
            // Immediate capture for dropdown/select-like inputs or when selection changes
            if (e.target.tagName === 'SELECT' || 
                e.inputType === 'deleteContentBackward' || 
                e.inputType === 'insertFromPaste' ||
                e.inputType === 'insertFromDrop') {
                captureInteraction('input_immediate', e, {
                    value: e.target.value || e.target.textContent,
                    inputType: e.inputType || null,
                    immediate: true
                });
            }
        }, true);
        
        // Text selection events
        document.addEventListener('select', (e) => {
            if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
                const selectedText = e.target.value.substring(e.target.selectionStart, e.target.selectionEnd);
                captureInteraction('select', e, {
                    selectedText: selectedText,
                    selectionStart: e.target.selectionStart,
                    selectionEnd: e.target.selectionEnd,
                    value: e.target.value,
                    selectionLength: selectedText.length
                });
            }
        }, true);
        
        // Clipboard events
        document.addEventListener('cut', (e) => {
            captureInteraction('cut', e, {
                clipboardData: e.clipboardData ? Array.from(e.clipboardData.types) : null,
                targetValue: e.target.value || e.target.textContent
            });
        }, true);
        
        document.addEventListener('copy', (e) => {
            captureInteraction('copy', e, {
                clipboardData: e.clipboardData ? Array.from(e.clipboardData.types) : null,
                targetValue: e.target.value || e.target.textContent
            });
        }, true);
        
        document.addEventListener('paste', (e) => {
            captureInteraction('paste', e, {
                clipboardData: e.clipboardData ? Array.from(e.clipboardData.types) : null,
                targetValue: e.target.value || e.target.textContent
            });
        }, true);
        
        // Enhanced form change events with better dropdown handling
        document.addEventListener('change', (e) => {
            let extra = {};
            if (e.target.tagName === 'SELECT') {
                const option = e.target.options[e.target.selectedIndex];
                extra = {
                    selectedValue: e.target.value,
                    selectedText: option?.text || '',
                    selectedIndex: e.target.selectedIndex,
                    allOptions: Array.from(e.target.options).map(opt => ({
                        value: opt.value,
                        text: opt.text,
                        selected: opt.selected
                    })),
                    optionsCount: e.target.options.length
                };
            } else if (['checkbox', 'radio'].includes(e.target.type)) {
                extra = {
                    checked: e.target.checked,
                    value: e.target.value,
                    name: e.target.name
                };
            } else {
                extra = {
                    value: e.target.value,
                    previousValue: e.target.defaultValue, // Capture what it was before
                    inputType: e.target.type
                };
            }
            captureInteraction('change', e, extra);
        }, true);
        
        document.addEventListener('submit', (e) => {
            captureInteraction('submit', e, {
                formAction: e.target.action || null,
                formMethod: e.target.method || 'GET',
                formElements: Array.from(e.target.elements).length
            });
        }, true);
        
        // Additional events for better field interaction capture
        
        // Option selection in datalists
        document.addEventListener('input', (e) => {
            if (e.target.list) { // Has datalist
                captureInteraction('datalist_input', e, {
                    value: e.target.value,
                    listId: e.target.list.id,
                    optionsCount: e.target.list.options.length
                });
            }
        }, true);
        
        // File input changes
        document.addEventListener('change', (e) => {
            if (e.target.type === 'file') {
                captureInteraction('file_select', e, {
                    filesCount: e.target.files.length,
                    files: Array.from(e.target.files).map(file => ({
                        name: file.name,
                        type: file.type,
                        size: file.size,
                        lastModified: file.lastModified
                    }))
                });
            }
        }, true);
    """


def get_scroll_event_listeners() -> str:
    """Get JavaScript code for scroll event listeners."""
    return """
        // Scroll events with debouncing to reduce noise
        let scrollTimeout;
        let lastScrollTime = 0;
        
        document.addEventListener('scroll', (e) => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                const now = Date.now();
                // Only capture scroll if it's been at least 200ms since last scroll capture
                if (now - lastScrollTime > 200) {
                    lastScrollTime = now;
                    captureInteraction('scroll', e, {
                        scrollX: window.scrollX,
                        scrollY: window.scrollY,
                        scrollLeft: e.target.scrollLeft || 0,
                        scrollTop: e.target.scrollTop || 0
                    });
                }
            }, 150); // Increased debounce time
        }, true);
        
        // Wheel events (for detailed scroll tracking) with throttling
        let lastWheelTime = 0;
        document.addEventListener('wheel', (e) => {
            const now = Date.now();
            // Only capture wheel events every 100ms to reduce noise
            if (now - lastWheelTime > 100) {
                lastWheelTime = now;
                captureInteraction('wheel', e, {
                    deltaX: e.deltaX,
                    deltaY: e.deltaY,
                    deltaZ: e.deltaZ,
                    deltaMode: e.deltaMode
                });
            }
        }, true);
    """


def get_focus_event_listeners() -> str:
    """Get JavaScript code for focus event listeners."""
    return """
        // Focus events - only for interactive elements to reduce noise
        document.addEventListener('focus', (e) => {
            // Only capture focus on interactive elements
            const interactiveElements = ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'];
            if (interactiveElements.includes(e.target.tagName) || 
                e.target.contentEditable === 'true' || 
                e.target.tabIndex >= 0) {
                captureInteraction('focus', e);
            }
        }, true);
        
        document.addEventListener('blur', (e) => {
            // Only capture blur on interactive elements
            const interactiveElements = ['INPUT', 'TEXTAREA', 'SELECT', 'BUTTON', 'A'];
            if (interactiveElements.includes(e.target.tagName) || 
                e.target.contentEditable === 'true' || 
                e.target.tabIndex >= 0) {
                captureInteraction('blur', e);
            }
        }, true);
    """


def get_recording_indicators_script() -> str:
    """Get JavaScript code for recording indicators."""
    return """
        // Remove any existing indicators
        const existingBorder = document.getElementById('__rec_border');
        if (existingBorder) existingBorder.remove();
        const existingIndicator = document.getElementById('__rec');
        if (existingIndicator) existingIndicator.remove();
        
        // Create border overlay
        const border = document.createElement('div');
        border.id = '__rec_border';
        border.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            border: 8px solid #ff0000;
            box-sizing: border-box;
            pointer-events: none;
            z-index: 999999;
            animation: pulse 1.5s infinite;
        `;
        
        // Create status indicator
        const indicator = document.createElement('div');
        indicator.id = '__rec';
        indicator.innerHTML = '🔴 RECORDING - Perform your action now';
        indicator.style.cssText = `
            position: fixed;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            background: #ff0000;
            color: #fff;
            padding: 12px 20px;
            border-radius: 8px;
            font: bold 10px -apple-system, BlinkMacSystemFont, sans-serif;
            z-index: 9999999;
            box-shadow: 0 4px 12px rgba(255,0,0,0.4);
            animation: pulse 1.5s infinite;
        `;
        
        // Add pulsing animation
        const style = document.createElement('style');
        style.textContent = `
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.4; }
                100% { opacity: 0.8; }
            }
        `;
        document.head.appendChild(style);
        
        document.body.appendChild(border);
        document.body.appendChild(indicator);
    """
