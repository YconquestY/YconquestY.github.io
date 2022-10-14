const plugins = {
    comment: {
        provider  : "Giscus",
        repo      : "YconquestY/comments", // `data-repo`
        repoId    : "R_kgDOIFj9NQ",        // `data-repo-id`
        category  : "General",             // `data-category`
        categoryId: "DIC_kwDOIFj9Nc4CRr3V" // `data-category-id`
    },
    components: ["Badge", "CodePen", "YouTube"],
    mdEnhance: {
        codetabs: true,     // enable code tabs
                            // custom container enabled by default
                            // `info` and `note` container UI exchanged?
        footnote: true,     // enable footnote
        mermaid: true,      // enable mermaid.js
        presentation: true, // enable presentation mode
        sub: true,          // enable subscripts
        sup: true,          // enable superscripts
        tabs: true,         // enable tabs
        katex: true         // enable KaTeX
    }
}

module.exports = plugins