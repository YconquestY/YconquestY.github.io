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
});