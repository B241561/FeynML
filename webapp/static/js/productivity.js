document.addEventListener('DOMContentLoaded', function() {
    if (window.location.pathname === '/notes') {
        initializeNotesPage();
    }
    if (window.location.pathname === '/tasks') {
        initializeTasksPage();
    }
});

function initializeNotesPage() {
    const noteForm = document.getElementById('note-form');
    const noteIdInput = document.getElementById('note_id');
    const titleInput = document.getElementById('note-title');
    const contentInput = document.getElementById('note-content');
    const preview = document.getElementById('note-preview-content');
    const status = document.getElementById('note-status');
    const searchInput = document.getElementById('note-search');
    const newNoteButton = document.getElementById('new-note');
    const noteCards = Array.from(document.querySelectorAll('.note-card'));
    let saveTimer = null;
    let lastDraft = '';

    function updatePreview() {
        const markdown = contentInput.value || '';
        if (window.marked) {
            preview.innerHTML = marked.parse(markdown);
        } else {
            preview.textContent = markdown;
        }
    }

    function setStatus(message, tone = 'text-muted') {
        status.textContent = message;
        status.className = `small ${tone}`;
    }

    function saveNote(auto = false) {
        const title = titleInput.value.trim();
        const content = contentInput.value.trim();
        if (!title || !content) {
            setStatus('Note requires both title and content.', 'text-muted');
            return;
        }

        const formData = new FormData(noteForm);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', noteForm.action, true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.onload = function() {
            if (xhr.status >= 200 && xhr.status < 300) {
                try {
                    const response = JSON.parse(xhr.responseText);
                    if (response.note_id) {
                        noteIdInput.value = response.note_id;
                    }
                    setStatus(auto ? 'Auto-saved' : 'Saved', 'text-success');
                    if (!noteIdInput.value || auto) {
                        window.location.reload();
                    }
                } catch (err) {
                    setStatus('Note saved.', 'text-success');
                }
            } else {
                setStatus('Unable to save note.', 'text-danger');
            }
        };
        xhr.onerror = function() {
            setStatus('Save request failed.', 'text-danger');
        };
        xhr.send(formData);
    }

    function scheduleAutoSave() {
        if (saveTimer) {
            clearTimeout(saveTimer);
        }
        saveTimer = setTimeout(function() {
            const title = titleInput.value.trim();
            const content = contentInput.value.trim();
            if (!title || !content) {
                setStatus('Autosave waiting for title and content...', 'text-muted');
                return;
            }
            saveNote(true);
        }, 1500);
    }

    function bindNoteCard(card) {
        const editButton = card.querySelector('.edit-note');
        const deleteButton = card.querySelector('.delete-note');

        if (editButton) {
            editButton.addEventListener('click', function() {
                const hiddenContent = card.querySelector('.note-full-content');
                titleInput.value = card.dataset.title || '';
                contentInput.value = hiddenContent ? hiddenContent.textContent.trim() : '';
                noteIdInput.value = card.dataset.noteId || '';
                updatePreview();
                setStatus('Editing note...', 'text-muted');
                titleInput.focus();
            });
        }
        if (deleteButton) {
            deleteButton.addEventListener('click', function() {
                if (!confirm('Delete this note permanently?')) {
                    return;
                }
                const noteId = card.dataset.noteId;
                fetch(`/notes/delete/${noteId}`, { method: 'POST', headers: { 'X-Requested-With': 'XMLHttpRequest' }})
                    .then(res => res.json())
                    .then(data => {
                        if (data.status === 'success') {
                            window.location.reload();
                        } else {
                            setStatus(data.message || 'Unable to delete note.', 'text-danger');
                        }
                    })
                    .catch(() => {
                        setStatus('Delete request failed.', 'text-danger');
                    });
            });
        }
    }

    if (noteCards.length) {
        noteCards.forEach(bindNoteCard);
    }

    searchInput.addEventListener('input', function() {
        const term = searchInput.value.trim().toLowerCase();
        noteCards.forEach(card => {
            const title = card.dataset.title.toLowerCase();
            const hiddenContent = card.querySelector('.note-full-content');
            const content = hiddenContent ? hiddenContent.textContent.toLowerCase() : '';
            card.style.display = title.includes(term) || content.includes(term) ? 'block' : 'none';
        });
    });

    contentInput.addEventListener('input', function() {
        updatePreview();
        scheduleAutoSave();
    });

    titleInput.addEventListener('input', function() {
        scheduleAutoSave();
    });

    noteForm.addEventListener('submit', function(event) {
        event.preventDefault();
        saveNote(false);
    });

    newNoteButton.addEventListener('click', function() {
        noteForm.reset();
        noteIdInput.value = '';
        updatePreview();
        setStatus('Ready to create a new note.', 'text-muted');
        titleInput.focus();
    });

    updatePreview();
    setStatus('Autosave enabled', 'text-muted');
}

function initializeTasksPage() {
    const taskForm = document.getElementById('task-form');
    const taskIdInput = document.getElementById('task_id');
    const titleInput = document.getElementById('task-title');
    const descriptionInput = document.getElementById('task-description');
    const prioritySelect = document.getElementById('task-priority');
    const dueDateInput = document.getElementById('task-due-date');
    const completedInput = document.getElementById('task-completed');
    const newTaskButton = document.getElementById('new-task');
    const taskCards = Array.from(document.querySelectorAll('.task-card'));

    function resetTaskForm() {
        taskForm.reset();
        taskIdInput.value = '';
    }

    function loadTask(card) {
        const hiddenDescription = card.querySelector('.task-full-description');
        taskIdInput.value = card.dataset.taskId || '';
        titleInput.value = card.dataset.title || '';
        descriptionInput.value = hiddenDescription ? hiddenDescription.textContent.trim() : '';
        prioritySelect.value = card.dataset.priority || 'Medium';
        dueDateInput.value = card.dataset.dueDate || '';
        completedInput.checked = card.dataset.completed === 'true';
        titleInput.focus();
    }

    function bindTaskCard(card) {
        const editButton = card.querySelector('.edit-task');
        const deleteButton = card.querySelector('.delete-task');
        const completeButton = card.querySelector('.complete-task');

        if (editButton) {
            editButton.addEventListener('click', function() {
                loadTask(card);
            });
        }
        if (deleteButton) {
            deleteButton.addEventListener('click', function() {
                if (!confirm('Remove this task?')) {
                    return;
                }
                const taskId = card.dataset.taskId;
                fetch(`/tasks/delete/${taskId}`, { method: 'POST', headers: { 'X-Requested-With': 'XMLHttpRequest' }})
                    .then(res => res.json())
                    .then(data => {
                        if (data.status === 'success') {
                            window.location.reload();
                        }
                    });
            });
        }
        if (completeButton) {
            completeButton.addEventListener('click', function() {
                const taskId = card.dataset.taskId;
                fetch(`/tasks/toggle/${taskId}`, { method: 'POST', headers: { 'X-Requested-With': 'XMLHttpRequest' }})
                    .then(res => res.json())
                    .then(data => {
                        if (data.status === 'success') {
                            window.location.reload();
                        }
                    });
            });
        }
    }

    if (taskCards.length) {
        taskCards.forEach(bindTaskCard);
    }

    taskForm.addEventListener('submit', function(event) {
        event.preventDefault();
        const formData = new FormData(taskForm);
        const xhr = new XMLHttpRequest();
        xhr.open('POST', taskForm.action, true);
        xhr.setRequestHeader('X-Requested-With', 'XMLHttpRequest');
        xhr.onload = function() {
            if (xhr.status >= 200 && xhr.status < 300) {
                window.location.reload();
            }
        };
        xhr.send(formData);
    });

    newTaskButton.addEventListener('click', function() {
        resetTaskForm();
        titleInput.focus();
    });
}
