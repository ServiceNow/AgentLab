# AgentLab Landing Page

This is a research landing page for AgentLab built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).

## Structure

```
docs/landing_page/
├── index.html              # Main landing page
├── projects/               # Individual project pages
│   ├── browsergym.html     # BrowserGym Ecosystem page
│   ├── webarena.html       # WebArena Evaluation page
│   └── workarena.html      # WorkArena Benchmark page
└── static/                 # Static assets
    ├── css/                # Stylesheets
    ├── js/                 # JavaScript files
    └── images/             # Images and icons
```

## Features

- **Responsive Design**: Built with Bulma CSS framework for mobile-friendly layouts
- **Project Navigation**: Dropdown menu linking to individual project pages
- **Academic Template**: Uses the popular academic project page template
- **Interactive Elements**: Smooth scrolling, animations, and hover effects
- **Multiple Projects**: Separate pages for BrowserGym, WebArena, and WorkArena
- **Social Media Ready**: Includes meta tags for social sharing

## Usage

### Viewing Locally

1. Open `index.html` in a web browser
2. Navigate between project pages using the dropdown menu
3. All links to external resources (GitHub, arXiv, etc.) are functional

### Hosting

This page can be hosted on:
- GitHub Pages
- Netlify
- Vercel
- Any static site hosting service

### Customization

1. **Update Content**: Edit the HTML files to update project information
2. **Add Images**: Replace placeholder images in `static/images/`
3. **Add Projects**: Create new HTML files in `projects/` directory
4. **Styling**: Modify `static/css/index.css` for custom styling

## Required Images

The following images should be added to `static/images/`:

1. **favicon.ico** - Site favicon (16x16 or 32x32 px)
2. **agentlab_overview.png** - Main overview diagram for landing page
3. **social_preview.png** - Social media preview image (1200x630 px)

Current placeholder images are provided as SVG files.

## Dependencies

The page uses CDN links for:
- Bulma CSS Framework
- FontAwesome Icons
- jQuery
- Academic Icons

No build process or installation required.

## Project Pages

### BrowserGym Ecosystem (`projects/browsergym.html`)
- Paper: https://arxiv.org/abs/2412.05467
- Code: https://github.com/ServiceNow/BrowserGym
- Focus: Unified web agent research framework

### WebArena Evaluation (`projects/webarena.html`)
- Website: https://webarena.dev/
- Setup: BrowserGym integration
- Focus: 812 realistic web tasks

### WorkArena Benchmark (`projects/workarena.html`)
- Repository: https://github.com/ServiceNow/WorkArena
- Focus: Enterprise-focused web agent evaluation
- Levels: L1 (33 tasks), L2/L3 (341 tasks each)

## Deployment

### GitHub Pages (Automatic)

The landing page is automatically deployed to GitHub Pages when changes are pushed to the main branch. The deployment is handled by the GitHub Actions workflow in `.github/workflows/deploy-landing-page.yml`.

**Setup Steps:**

1. Go to your GitHub repository settings
2. Navigate to "Pages" in the left sidebar
3. Under "Source", select "GitHub Actions"
4. The site will be available at: `https://[username].github.io/AgentLab/`

**Manual Trigger:**

You can manually trigger the deployment by going to the "Actions" tab in your GitHub repository and running the "Deploy Landing Page to GitHub Pages" workflow.

### Local Development Server

For local testing:

```bash
cd docs/landing_page
python3 -m http.server 8000
# Visit http://localhost:8000
```

## Contributing

To add a new project page:

1. Create a new HTML file in `projects/` directory
2. Use existing project pages as templates
3. Update the dropdown menu in `index.html`
4. Add a project card to the main landing page
5. Include appropriate links and metadata

## License

This template follows the Academic Project Page Template license (Creative Commons Attribution-ShareAlike 4.0 International License).

AgentLab is developed by ServiceNow Research and follows its respective licensing terms.