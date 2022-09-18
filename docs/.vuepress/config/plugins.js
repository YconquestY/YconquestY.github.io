const plugins = {
    mdEnhance: {
        codetabs: true,     // enable code tabs
                            // custom container enabled by default
                            // `info` and `note` container UI exchanged?
        mermaid: true,      // enable mermaid.js
        presentation: true, // enable presentation mode
        sub: true,          // enable subscripts
        sup: true,          // enable superscripts
        tabs: true,         // enable tabs
        tex: true,          // patch
        katex: true         // enable KaTeX
    },
    components: ["Badge", "CodePen", "YouTube"]
}

module.exports = plugins