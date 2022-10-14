import{_ as r}from"./_plugin-vue_export-helper.cdc0426e.js";import{o as c,c as m,a as h,b as s,d as t,w as o,e as p,f as a,r as n}from"./app.58949572.js";const d="/assets/me.c7c812f8.jpg",u="/assets/ndl.9339fc6f.png",_="/assets/ds_distillation.3dbd4b2e.png",g={},y=p('<h1 id="hello-there" tabindex="-1"><a class="header-anchor" href="#hello-there" aria-hidden="true">#</a> Hello there</h1><img src="'+d+'" alt="Me" title="copyright \xA9 YU Yue" width="300px"><h2 id="about-me" tabindex="-1"><a class="header-anchor" href="#about-me" aria-hidden="true">#</a> About me</h2>',3),b=a("I am YU Yue (Will), an undergraduate at the "),f={href:"https://www.hku.hk",target:"_blank",rel:"noopener noreferrer"},x=a("University of Hong Kong"),w=a("."),k=s("h2",{id:"research",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#research","aria-hidden":"true"},"#"),a(" Research")],-1),M=s("p",null,[a("My research interest includes machine learning system, neural rendering, "),s("strong",null,"in-memory computing"),a(", and computer architecture.")],-1),v=s("h3",{id:"nerf",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#nerf","aria-hidden":"true"},"#"),a(" NeRF")],-1),N={href:"https://www.matthewtancik.com/nerf",target:"_blank",rel:"noopener noreferrer"},L=s("em",null,"Neural radiance field",-1),R=a(" (NeRF) "),F=s("strong",null,"implicitly",-1),I=a(" represents a 3D scene with a multi-layer perceptron (MLP) "),T=s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"F"),s("mo",null,":"),s("mo",{stretchy:"false"},"("),s("mi",{mathvariant:"bold-italic"},"x"),s("mo",{separator:"true"},","),s("mi",{mathvariant:"bold-italic"},"d"),s("mo",{stretchy:"false"},")"),s("mo",null,"\u2192"),s("mo",{stretchy:"false"},"("),s("mi",{mathvariant:"bold-italic"},"c"),s("mo",{separator:"true"},","),s("mi",null,"\u03C3"),s("mo",{stretchy:"false"},")")]),s("annotation",{encoding:"application/x-tex"},"F: (\\boldsymbol{x}, \\boldsymbol{d}) \\rightarrow (\\boldsymbol{c}, \\sigma)")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6833em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.13889em"}},"F"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},":"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mopen"},"("),s("span",{class:"mord"},[s("span",{class:"mord"},[s("span",{class:"mord boldsymbol"},"x")])]),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord"},[s("span",{class:"mord"},[s("span",{class:"mord boldsymbol"},"d")])]),s("span",{class:"mclose"},")"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"\u2192"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mopen"},"("),s("span",{class:"mord"},[s("span",{class:"mord"},[s("span",{class:"mord boldsymbol"},"c")])]),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"\u03C3"),s("span",{class:"mclose"},")")])])],-1),Y=a(" for some position "),C=s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",{mathvariant:"bold-italic"},"x"),s("mo",null,"\u2208"),s("msup",null,[s("mi",{mathvariant:"double-struck"},"R"),s("mn",null,"3")])]),s("annotation",{encoding:"application/x-tex"},"\\boldsymbol{x} \\in \\mathbb{R}^3")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.5782em","vertical-align":"-0.0391em"}}),s("span",{class:"mord"},[s("span",{class:"mord"},[s("span",{class:"mord boldsymbol"},"x")])]),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"\u2208"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.8141em"}}),s("span",{class:"mord"},[s("span",{class:"mord mathbb"},"R"),s("span",{class:"msupsub"},[s("span",{class:"vlist-t"},[s("span",{class:"vlist-r"},[s("span",{class:"vlist",style:{height:"0.8141em"}},[s("span",{style:{top:"-3.063em","margin-right":"0.05em"}},[s("span",{class:"pstrut",style:{height:"2.7em"}}),s("span",{class:"sizing reset-size6 size3 mtight"},[s("span",{class:"mord mtight"},"3")])])])])])])])])])],-1),D=a(", view direction "),U=s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",{mathvariant:"bold-italic"},"d"),s("mo",null,"\u2208"),s("mo",{stretchy:"false"},"["),s("mn",null,"0"),s("mo",{separator:"true"},","),s("mi",null,"\u03C0"),s("mo",{stretchy:"false"},")"),s("mo",null,"\xD7"),s("mo",{stretchy:"false"},"["),s("mn",null,"0"),s("mo",{separator:"true"},","),s("mn",null,"2"),s("mi",null,"\u03C0"),s("mo",{stretchy:"false"},")")]),s("annotation",{encoding:"application/x-tex"},"\\boldsymbol{d} \\in [0, \\pi) \\times [0, 2\\pi)")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.7335em","vertical-align":"-0.0391em"}}),s("span",{class:"mord"},[s("span",{class:"mord"},[s("span",{class:"mord boldsymbol"},"d")])]),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}}),s("span",{class:"mrel"},"\u2208"),s("span",{class:"mspace",style:{"margin-right":"0.2778em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mopen"},"["),s("span",{class:"mord"},"0"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"\u03C0"),s("span",{class:"mclose"},")"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}}),s("span",{class:"mbin"},"\xD7"),s("span",{class:"mspace",style:{"margin-right":"0.2222em"}})]),s("span",{class:"base"},[s("span",{class:"strut",style:{height:"1em","vertical-align":"-0.25em"}}),s("span",{class:"mopen"},"["),s("span",{class:"mord"},"0"),s("span",{class:"mpunct"},","),s("span",{class:"mspace",style:{"margin-right":"0.1667em"}}),s("span",{class:"mord"},"2"),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"\u03C0"),s("span",{class:"mclose"},")")])])],-1),z=a(", color "),V=s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",{mathvariant:"bold-italic"},"c")]),s("annotation",{encoding:"application/x-tex"},"\\boldsymbol{c}")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.4444em"}}),s("span",{class:"mord"},[s("span",{class:"mord"},[s("span",{class:"mord boldsymbol"},"c")])])])])],-1),q=a(', and "opacity" '),E=s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mi",null,"\u03C3")]),s("annotation",{encoding:"application/x-tex"},"\\sigma")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.4306em"}}),s("span",{class:"mord mathnormal",style:{"margin-right":"0.03588em"}},"\u03C3")])])],-1),K=a(". Rendered results are spectacular."),P=a("I am currently working on rapid and accurate 3D reconstruction of real-world scenes with NeRF. See "),S=a("my posts"),j=a(" ("),B=a("link"),G=a(") on NeRF."),W=s("h3",{id:"mlsys",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#mlsys","aria-hidden":"true"},"#"),a(" MLSys")],-1),A=s("img",{src:u,alt:"needle",title:"source: 10-414/714 by CMU"},null,-1),H=s("em",null,"Needle",-1),J=a(" is a deep learning library with customized GPU and NumPy CPU backend. The project is still in progress; see "),Q={href:"https://github.com/YconquestY/Needle",target:"_blank",rel:"noopener noreferrer"},X=a("my repository"),O=a(" for details."),Z=s("h3",{id:"ntk",tabindex:"-1"},[s("a",{class:"header-anchor",href:"#ntk","aria-hidden":"true"},"#"),a(" NTK")],-1),$=s("img",{src:_,alt:"Dataset distillation",title:"source: Dataset Distillation with Infinitely Wide Convolutional Networks by Timothy Nguyen, Roman Novak, Lechao Xiao, and Jaehoon Lee"},null,-1),ss=s("p",null,[a("Ever though of achieving "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mn",null,"65")]),s("annotation",{encoding:"application/x-tex"},"65")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6444em"}}),s("span",{class:"mord"},"65")])])]),a("% test accuracy on CIFAR-10 with merely "),s("span",{class:"katex"},[s("span",{class:"katex-mathml"},[s("math",{xmlns:"http://www.w3.org/1998/Math/MathML"},[s("semantics",null,[s("mrow",null,[s("mn",null,"10")]),s("annotation",{encoding:"application/x-tex"},"10")])])]),s("span",{class:"katex-html","aria-hidden":"true"},[s("span",{class:"base"},[s("span",{class:"strut",style:{height:"0.6444em"}}),s("span",{class:"mord"},"10")])])]),a(" data points? "),s("em",null,"Neural tangent kernel"),a(" (NTK) is the way."),s("sup",{class:"footnote-ref"},[s("a",{href:"#footnote1"},"[1]"),s("a",{class:"footnote-anchor",id:"footnote-ref1"})])],-1),as=s("hr",{class:"footnotes-sep"},null,-1),ts={class:"footnotes"},es={class:"footnotes-list"},ns={id:"footnote1",class:"footnote-item"},ls=a('"This is the way" is a line in the Star Wars series '),os={href:"https://www.disneyplus.com/en-gb/series/the-mandalorian/3jLIGMDYINqD",target:"_blank",rel:"noopener noreferrer"},is=a("the Mandalorian"),rs=a(', which describes the "Mando" style. It is cited here to emphasize only NTK may achieve such an unbelievable result as dataset distillation. '),cs=s("a",{href:"#footnote-ref1",class:"footnote-backref"},"\u21A9\uFE0E",-1);function ms(hs,ps){const e=n("ExternalLinkIcon"),i=n("YouTube"),l=n("RouterLink");return c(),m("div",null,[h(" landing page "),y,s("p",null,[b,s("a",f,[x,t(e)]),w]),k,M,v,t(i,{id:"gGaqqs5Q-yo",autoplay:"true",loop:"true",disableFullscreen:"false"}),s("p",null,[s("a",N,[L,t(e)]),R,F,I,T,Y,C,D,U,z,V,q,E,K]),s("p",null,[P,t(l,{to:"/blog/ml/nerf/"},{default:o(()=>[S]),_:1}),j,t(l,{to:"/blog/ml/nerf/"},{default:o(()=>[B]),_:1}),G]),W,A,s("p",null,[H,J,s("a",Q,[X,t(e)]),O]),Z,$,ss,as,s("section",ts,[s("ol",es,[s("li",ns,[s("p",null,[ls,s("a",os,[is,t(e)]),rs,cs])])])])])}const _s=r(g,[["render",ms],["__file","index.html.vue"]]);export{_s as default};
