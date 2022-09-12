import { hopeTheme } from "vuepress-theme-hope"

const head = require('./config/head')
//const nav  = require('./config/nav')
const plugins = require("./config/plugins")


export default {
    title: 'Will\'s Blog',
    description: 'Hello there.', // welcoming message by Obi-Wan Kenobi
    head: head, // do not use `this.head`
    markdown: {
       code: {
           lineNumbers: false // hide line index
       }
    },
    theme: hopeTheme({
        plugins: plugins
    })
}