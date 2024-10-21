import { hopeTheme } from "vuepress-theme-hope";

import navbar from "./config/navbar.js";
import sidebar from "./config/sidebar.js";
import plugins from "./config/plugins.js";

export default hopeTheme({
    hostname: "https://yconquesty.github.io",

    author: {
        name: "Yue Yu",
        url: "https://yconquesty.github.io",
        email: "yconquesty@gmail.com"
    },
    license: "CC BY-NC-SA 4.0",

    logo: "logo.svg",
    logoDark: "logoDark.svg",
    favicon: "git.svg", // TODO

    // navigation bar
    navbar,
    navbarTitle: "Yue Yu",

    repo: "YconquestY/YconquestY.github.io",
    repoLabel: "GitHub",
    
    sidebar,

    prevLink: false,
    nextLink: false,

    pageInfo: ["Date", "Category", "Tag"],
    breadcrumb: false,

    lastUpdated: false,
    contributors: false,
    editLink: false,
    docsDir: "docs",

    //footer: ,
    copyright: "Copyright © 2024 Yue Yu",
    displayFooter: true,

    iconAssets: "fontawesome-with-brands",

    // must enable blog plugin
    blog: {
        avatar: "img/me.png",
        name: "Yue Yu",
        description: "Computer systems and architecture",
        intro: "/about.html",
        medias: {
            //Instagram: ,
            //Wechat: ,
            //Weibo: ,
            //Whatsapp: ,

            Email: "mailto:yconquesty@gmail.com",
            //Gmail: ,
            GitHub: "https://github.com/YconquestY",
            Linkedin: "https://www.linkedin.com/in/yue-yu-4609b8197/"
            
            //Xiaohongshu: ,
            //Zhihu: ,

            //Youtube: ,
        }
    },

    //hotReload: true,

    plugins
});
