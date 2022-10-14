import { hopeTheme } from "vuepress-theme-hope"

//const nav = require('./config/nav')
const plugins = require("./config/plugins")


export default {
    /* I18n */
    lang: "en-US",
    /* theme */
    author: "YU Yue",
    /* layout */
    //navbar: ?
    logo: "./logo.png",
    repo: "https://github.com/YconquestY/YconquestY.github.io",
    repoLabel: "GitHub",
    repoDisplay: true,
    //sidebar: ?
    prevLink: false,
    nextLink: false,
    pageInfo: ["Author", "Original"],
    lastUpdated: false,
    contributors: false,
    editLink: false,
    copyright: "Copyright © YU Yue",
    title: "Will",
    description: 'YU Yue\'s homepage',
    markdown: {
       code: {
           lineNumbers: false // hide line index
       }
    },
    theme: hopeTheme({
        plugins: plugins
    })
}