---
name: Aura & Ink
colors:
  surface: '#faf9f7'
  surface-dim: '#dadad8'
  surface-bright: '#faf9f7'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f4f3f1'
  surface-container: '#efeeec'
  surface-container-high: '#e9e8e6'
  surface-container-highest: '#e3e2e0'
  on-surface: '#1a1c1b'
  on-surface-variant: '#4c4546'
  inverse-surface: '#2f3130'
  inverse-on-surface: '#f1f1ef'
  outline: '#7e7576'
  outline-variant: '#cfc4c5'
  surface-tint: '#5e5e5e'
  primary: '#000000'
  on-primary: '#ffffff'
  primary-container: '#1b1b1b'
  on-primary-container: '#848484'
  inverse-primary: '#c6c6c6'
  secondary: '#515f74'
  on-secondary: '#ffffff'
  secondary-container: '#d2e1fa'
  on-secondary-container: '#556379'
  tertiary: '#000000'
  on-tertiary: '#ffffff'
  tertiary-container: '#0b1c30'
  on-tertiary-container: '#75859d'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#e2e2e2'
  primary-fixed-dim: '#c6c6c6'
  on-primary-fixed: '#1b1b1b'
  on-primary-fixed-variant: '#474747'
  secondary-fixed: '#d5e3fc'
  secondary-fixed-dim: '#b9c7e0'
  on-secondary-fixed: '#0d1c2e'
  on-secondary-fixed-variant: '#3a485c'
  tertiary-fixed: '#d3e4fe'
  tertiary-fixed-dim: '#b7c8e1'
  on-tertiary-fixed: '#0b1c30'
  on-tertiary-fixed-variant: '#38485d'
  background: '#faf9f7'
  on-background: '#1a1c1b'
  surface-variant: '#e3e2e0'
typography:
  display-xl:
    fontFamily: Syne
    fontSize: 84px
    fontWeight: '800'
    lineHeight: 92px
    letterSpacing: -0.04em
  headline-lg:
    fontFamily: Syne
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Syne
    fontSize: 36px
    fontWeight: '700'
    lineHeight: 44px
    letterSpacing: -0.02em
  body-lg:
    fontFamily: Hanken Grotesk
    fontSize: 20px
    fontWeight: '400'
    lineHeight: 32px
    letterSpacing: 0em
  body-md:
    fontFamily: Hanken Grotesk
    fontSize: 16px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  label-sm:
    fontFamily: JetBrains Mono
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.05em
rounded:
  sm: 0.125rem
  DEFAULT: 0.25rem
  md: 0.375rem
  lg: 0.5rem
  xl: 0.75rem
  full: 9999px
spacing:
  baseline: 8px
  container-max: 1100px
  gutter: 32px
  section-gap-desktop: 160px
  section-gap-mobile: 80px
  edge-margin-mobile: 20px
---

## Brand & Style
The design system is rooted in the philosophy of **Modern Editorial Minimalism**. It moves away from high-intensity "tech" aesthetics (glows, neon, complex grids) to position the user as an architect of systems—intellectual, refined, and quietly confident. 

The aesthetic is characterized by:
- **Intellectual Tension:** The interplay between massive, avant-garde typography and generous, intentional whitespace.
- **Print-Inspired Layouts:** Asymmetrical compositions that mimic high-fashion journals or premium broadsheets.
- **Quiet Professionalism:** A focus on legibility and narrative flow over frenetic interaction.
- **Premium Tangibility:** Every element is deliberate; decoration is achieved through scale and empty space rather than ornamentation.

## Colors
This design system employs a high-contrast palette centered on warmth and technical clarity.

- **Light Mode (Default):** Utilizes a "Paper" foundation (`#FAF9F7`) to provide a tactile, editorial feel and reduce the eye strain associated with pure white.
- **Dark Mode:** Transitions to "Inkwell" (`#121212`), utilizing soft white text (`#E5E7EB`) to maintain premium legibility.
- **Primary & Secondary:** A strict black/white core is accented by **Slate Blue** for interactive cues. This color is intentionally desaturated to remain harmonious with the neutral base.
- **Borders:** Use low-contrast `border_muted` values to define structure without creating visual noise.

## Typography
The typographic strategy uses drastic scale shifts to create hierarchy.

- **Display & Headlines:** Use **Syne** set with tight letter-spacing to create a "blocky," architectural effect. This font serves as the primary visual hook.
- **Body Text:** Use **Hanken Grotesk** for long-form content. Line heights are purposefully generous (1.5x+) to facilitate a comfortable reading experience in "Process" and "About" sections.
- **Technical Metadata:** Use **JetBrains Mono** for labels, tags, and small UI details to subtly reference the technical nature of the backend engineer's craft.

## Layout & Spacing
The layout follows a **Fixed Grid** philosophy for desktop and a **Fluid** model for mobile.

- **Desktop (1200px+):** A 12-column grid within a 1100px max-width container. Content should be centered with significant "breathing room" on the flanks.
- **Vertical Rhythm:** Use aggressive `section-gap-desktop` (160px) spacing between major sections to force a mental reset during scrolling.
- **Asymmetry:** Content should frequently "break" the grid. For example, a headline may span 8 columns while the remaining 4 columns are left entirely empty.
- **Mobile Adaptivity:** Scale section gaps down to 80px and margins to 20px. Headlines must transition to `headline-lg-mobile` to prevent awkward word wrapping.

## Elevation & Depth
This design system avoids traditional shadows to maintain its editorial integrity.

- **Tonal Layering:** Depth is suggested by subtle shifts in background color. Surfaces and containers use tiered colors (e.g., `surface-container`) to create stackable depth without lifting elements off the page.
- **Low-Contrast Outlines:** Use 1px "hairline" borders in `border-muted`. These are used for structural definition only.
- **Backdrop Blurs:** The navigation bar is the only element allowed to use glassmorphism. Use a high-blur (20px) and low-opacity (80%) backdrop filter to allow underlying content to peak through elegantly.
- **Interactions:** Avoid elevation/lifting on hover. Instead, use internal fills or subtle color shifts to indicate interactivity.

## Shapes
The shape language is **Soft-Modern**, balancing brutalist precision with approachable softness.

- **Standard UI Elements:** Use `rounded` (0.25rem) for buttons, input fields, and standard cards to take the "edge" off the grid.
- **Imagery:** Large project images and hero photographs must remain **Sharp (0px radius)** to mimic the appearance of a printed magazine.
- **Tags & Indicators:** Small labels or technical chips may use **Pill-shaped** (rounded-full) corners to distinguish them from structural UI blocks.

## Components

### Buttons
- **Primary:** Solid black (`primary_color_hex`) with white text. Wide proportions with generous horizontal padding.
- **Secondary:** Outlined with 1px `border_muted`.
- **Interaction:** Sharp, instant color transitions on hover; no shadow or scale effects.

### Navigation
A minimalist top-bar or "Floating Dock." Highly transparent with a heavy backdrop blur. Links are set in `label-sm` with a simple 1px underline transition.

### Project Blocks
Large-scale blocks featuring a hero image (sharp corners). The project title is set in `headline-lg`, followed by a short description in `body-md`. These should alternate alignment to maintain the editorial rhythm.

### Input Fields
Minimalist bottom-border only or a very light 1px outline. Labels use `label-sm` and sit above the field. Focus states shift the border color to `tertiary_color_hex` (Slate Blue).

### Chips / Tags
Used for technical stacks. Small, `label-sm` text inside a subtle gray fill. No borders. Keep spacing tight to ensure they appear as metadata rather than primary actions.

### Editorial Pull-Quotes
Large text snippets using `display-xl` to break up long-form body copy. Use horizontal rules (`<hr>`) to separate narrative sections.