import { defineUserConfig } from "vuepress" // necessary?
import { hopeTheme } from "vuepress-theme-hope"

//const nav = require('./config/nav')
const plugins = require("./config/plugins")


export default defineUserConfig({
    /* I18n */
    lang: "en-US",
    /* theme */
    author: "Yue Yu",
    /* layout */
    //navbar: ?
    logo: "./logo.png",
    repo: "https://github.com/YconquestY/YconquestY.github.io",
    //sidebar: ?
    prevLink: false,
    nextLink: false,
    pageInfo: ["Author", "Original"],
    lastUpdated: false,
    contributors: false,
    editLink: false,
    copyright: "Copyright © Yue Yu",
    title: "Will",
    description: 'Yue Yu (Will)\'s Blog',
    markdown: {
       code: {
           lineNumbers: false // hide line index
       }
    },
    theme: hopeTheme({
        plugins: plugins
    })
})