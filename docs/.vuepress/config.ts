import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
    base: "/",

    lang: "en-US", // no I18n
    title: "Yue Yu",
    description: "Yue Yu's website",

    theme,
    // enable PWA
    //shouldPrefetch: false,

    head: [
        ["link", {
            rel: "preconnect",
            href: "https://fonts.googleapis.com"
        }],
        ["link", {
            rel: "preconnect",
            href: "https://fonts.gstatic.com",
            crossorigin: ""
        }],
        ["link", {
            rel: "stylesheet",
            href: "https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap"
        }]
    ]
});