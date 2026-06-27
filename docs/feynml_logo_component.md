# FeynML JARVIS Logo Component

## Included files

- `webapp/templates/components/feynml_logo.html`
- `webapp/static/css/feynml-logo.css`
- `webapp/static/js/feynml-logo.js`

## Component API

Use the Jinja macro:

```jinja2
{% from "components/feynml_logo.html" import feynml_logo %}

{{ feynml_logo(
    instance_id='hero-logo',
    size='large',
    animated=true,
    theme='auto',
    show_wordmark=true,
    hover_effects=true,
    intro_animation=true
) }}
```

## Options

- `size`: `small | medium | large`
- `animated`: `true | false`
- `theme`: `auto | dark | light`
- `show_wordmark`: `true | false`
- `hover_effects`: `true | false`
- `intro_animation`: `true | false`
- `class_name`: extra CSS class string

## Live integrations

- Navbar: `webapp/templates/base.html`
- Footer: `webapp/templates/base.html`
- Landing hero: `webapp/templates/landing.html`
- Login screen: `webapp/templates/login.html`
- Loading screen: `webapp/templates/analysis_running.html`
- Dashboard header: `webapp/templates/dashboard.html`
- Report cover/header: `webapp/templates/report.html`

## Example usage

### Landing page

```jinja2
{{ feynml_logo(
    instance_id='landing-hero-logo',
    size='large',
    theme='dark',
    animated=true,
    show_wordmark=true,
    hover_effects=true,
    intro_animation=true,
    class_name='justify-content-center w-100'
) }}
```

### Navbar

```jinja2
{{ feynml_logo(
    instance_id='navbar-logo',
    size='small',
    theme='auto',
    animated=true,
    show_wordmark=true,
    hover_effects=true,
    intro_animation=false,
    class_name='navbar-brand-logo'
) }}
```

### Dashboard

```jinja2
{{ feynml_logo(
    instance_id='dashboard-header-logo',
    size='small',
    theme='auto',
    animated=true,
    show_wordmark=true,
    hover_effects=true,
    intro_animation=false,
    class_name='dashboard-header-logo'
) }}
```

### Loading screen

```jinja2
{{ feynml_logo(
    instance_id='loading-logo',
    size='large',
    theme='auto',
    animated=true,
    show_wordmark=true,
    hover_effects=false,
    intro_animation=true,
    class_name='justify-content-center'
) }}
```

### Empty state

```jinja2
<div class="empty-state">
    {{ feynml_logo(
        instance_id='empty-state-logo',
        size='medium',
        theme='auto',
        animated=false,
        show_wordmark=true,
        hover_effects=false,
        intro_animation=false,
        class_name='justify-content-center mb-3'
    ) }}
    <p class="text-muted mb-0">No diagnostics available yet.</p>
</div>
```

## JavaScript controller

The controller is globally available as `window.FeynMLLogo`.

```javascript
window.FeynMLLogo.mount('#custom-logo', {
    size: 'medium',
    animated: true,
    theme: 'dark',
    showWordmark: true,
    hoverEffects: true,
    introAnimation: true
});
```

## Performance notes

- All continuous motion uses CSS transforms and opacity instead of layout-heavy SVG attribute animation.
- Animation pauses automatically when the document is hidden.
- `prefers-reduced-motion` disables intro and looped effects.
- Static mode can be enabled per instance with `animated=false`.
- Hover acceleration is limited to pointer devices only.

## Accessibility notes

- Every logo instance renders with `role="img"`.
- Each instance gets a configurable `title` and `desc`.
- `instance_id` should remain unique per page to avoid duplicate ARIA IDs.
- The visible wordmark stays in HTML for crisp text rendering and better responsive control.
