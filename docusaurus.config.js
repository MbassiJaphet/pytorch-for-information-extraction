module.exports = {
  title: 'Pytorch For Information Extraction',
  tagline: 'Reliable open-source solution for Information Extraction',
  url: 'https://mbassijaphet.github.io/pytorch-for-information-extraction/',
  baseUrl: '/pytorch-for-information-extraction/',
  onBrokenLinks: 'throw',
  favicon: 'img/favicon.ico',
  organizationName: 'MbassiJaphet', // Usually your GitHub org/user name.
  projectName: 'pytorch-for-information-extraction', // Usually your repo name.
  /*
  plugins: [
    [
      '@docusaurus/plugin-content-docs',
      {
        id: 'tutorial',
        path: 'tutorial',
        editUrl: 'https://github.com/MbassiJaphet/pytorch-for-information-extraction/edit/master/website/',
        routeBasePath: '/',
        sidebarPath: require.resolve('./tutorial/sidebars.js'),
        showLastUpdateAuthor: true,
        showLastUpdateTime: true,
      },
    ],
  ],
  */
  themeConfig: {
    navbar: {
      hideOnScroll: false,
      title: 'Pytorch For Information Extraction',
      logo: {
        alt: 'My Site Logo',
        src: 'img/logo.svg',
      },
      items: [
        /*
        {
          to: 'docs/',
          activeBasePath: 'docs',
          label: 'Docs',
          position: 'left',
        },
        */
        {
          to: '/',
          label: 'Demo',
          position: 'right',
          className: 'demo-button'
        }
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Get Started',
          items: [
            {
              label: 'Style Guide',
              to: '/introduction',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Stack Overflow',
              href: 'https://stackoverflow.com/questions/tagged/pytorch-for-information-extraction',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/MbassiJaphet/pytorch-for-information-extraction',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Pytorch For Information Extraction. Built with Docusaurus.`,
    },
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: require.resolve('./docs/sidebars.js'),
          // Please change this to your repo.
          editUrl:
            'https://github.com/MbassiJaphet/pytorch-for-information-extraction/edit/master/website/',
        },
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      },
    ],
  ],
};
