import { PluginsOptions } from "vuepress-theme-hope";

const plugins: PluginsOptions = {
    markdownImage: {
        figure: true,
        lazyload: true,
        size: true
    },
    markdownMath: {
        type: "katex"
    },
    markdownHint: { // for hint container
        hint: true
    },
    markdownTab: {
        tabs: true,
    },
    mdEnhance: {
        //component: true, // markdown component; different from components plugin

        sup: true, // superscript
        sub: true, // subscript
        tasklist: true,

        footnote: true,
        include: true, // include files

        attrs: true, // attach HTML attributes to markdown
        align: true, // align to left/center/right; may be buggy
        mark: true, // hightlight text background
        spoiler: true, // hide text
        //stylize: [],

        //flowchart: true, // unlike mermaid flowchart for software architecture
        markmap: true, // mindmap, particularly tree diagram
        mermaid: true,
    },
    revealjs: { // for presentation
        plugins: ["highlight", "math", "zoom"]
    },
    blog: {
        filter: (page) => {
            // not shown on the home page by default
            if (page.frontmatter.article == true) {
                return true;
            } else {
                return false;
            }
        }
    },
    comment: {
        provider: "Giscus",
        repo: "YconquestY/comments",
        repoId: "R_kgDOIFj9NQ",
        category: "General",
        categoryId: "DIC_kwDOIFj9Nc4CRr3V"
    },
    components: { // different from `mdEnhance` components
        components: [
            "Badge", // badge at superscript position
            //"PDF",
            "Share",
            //"SiteInfo",
            //"VPBanner",
            //"VPCard",
            "VidStack" // for (YouTube) video and audio
        ]
    },
    readingTime: false
}

export default plugins;
